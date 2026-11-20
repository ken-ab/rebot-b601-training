"""B601 ACT inference using only PyTorch/Torchvision, NumPy, Pillow and Einops."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch

from act_network import ACT

CAMERAS = ('observation.images.overhead', 'observation.images.wrist')
JOINTS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos',
          'wrist_yaw.pos', 'wrist_roll.pos', 'gripper.pos')


def letterbox_rgb(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected HWC or CHW RGB')
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Float RGB must be finite in[0,1]')
        image = np.rint(image*255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError('RGB must be uint8 or float[0,1]')
    height, width = image.shape[:2]
    ratio = max(width/224, height/224)
    resized_h, resized_w = int(height/ratio), int(width/ratio)
    resized = np.asarray(Image.fromarray(image).resize((resized_w, resized_h), Image.Resampling.BILINEAR))
    result = np.zeros((224, 224, 3), dtype=np.uint8)
    top, left = (224-resized_h)//2, (224-resized_w)//2
    result[top:top+resized_h, left:left+resized_w] = resized
    return np.ascontiguousarray(result.transpose(2, 0, 1))


def namespace_config(raw):
    raw = dict(raw)
    raw['pretrained_backbone_weights'] = None  # Complete trained weights are local.
    raw['robot_state_feature'] = SimpleNamespace(shape=tuple(raw['input_features']['observation.state']['shape']))
    raw['action_feature'] = SimpleNamespace(shape=tuple(raw['output_features']['action']['shape']))
    raw['env_state_feature'] = None
    raw['image_features'] = {key: SimpleNamespace(shape=tuple(raw['input_features'][key]['shape'])) for key in CAMERAS}
    return SimpleNamespace(**raw)


class B601ACT:
    def __init__(self, directory=None, device=None):
        self.directory = Path(directory or Path(__file__).resolve().parent)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        raw = json.loads((self.directory/'config.json').read_text())
        self.model = ACT(namespace_config(raw))
        weights = torch.load(self.directory/'weights.pt', map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights['model'], strict=True)
        if not all(torch.isfinite(value).all().item() for value in self.model.state_dict().values()):
            raise ValueError('Model contains nonfinite parameters')
        self.model.to(self.device).eval()
        self.norm = {key: value.to(self.device) for key,value in weights['normalizers'].items()}
        if not all(torch.isfinite(value).all().item() for value in self.norm.values()):
            raise ValueError('Normalizer contains nonfinite values')
        self.metadata = {'robot_type': 'seeed_b601_rs_follower', 'policy': 'ACT', 'fps': 25,
            'action_horizon': 25, 'action_names': list(JOINTS), 'action_space': 'absolute_adapter_input',
            'language_conditioned': False, 'device': str(self.device)}

    @torch.inference_mode()
    def predict_batch(self, images, states):
        images = np.asarray(images)
        states = np.asarray(states, dtype=np.float32)
        if images.shape[1:] != (2, 3, 224, 224) or states.shape != (len(images), 7):
            raise ValueError('Expectedimages[B,2,3,224,224],states[B,7]')
        if images.dtype != np.uint8 or not np.isfinite(states).all():
            raise ValueError('Expecteduint8images andfinite rawstates')
        image_tensor = torch.from_numpy(np.ascontiguousarray(images)).to(self.device).float()/255
        state_tensor = torch.from_numpy(np.ascontiguousarray(states)).to(self.device)
        batch = {'observation.state': (state_tensor-self.norm['normalize_inputs.buffer_observation_state.mean']) /
                  (self.norm['normalize_inputs.buffer_observation_state.std']+1e-8)}
        batch['observation.images'] = []
        for index, key in enumerate(CAMERAS):
            prefix = 'normalize_inputs.buffer_'+key.replace('.', '_')
            batch['observation.images'].append((image_tensor[:,index]-self.norm[prefix+'.mean']) /
                                                (self.norm[prefix+'.std']+1e-8))
        # No action target/padding enters this call: ACT uses zero latent at deployment.
        normalized, latent = self.model(batch)
        if latent != (None, None):
            raise ValueError('Deployment must not use the VAE target posterior')
        actions = normalized*self.norm['unnormalize_outputs.buffer_action.std'] + self.norm['unnormalize_outputs.buffer_action.mean']
        result = actions.cpu().numpy()
        if result.shape != (len(images), 25, 7) or not np.isfinite(result).all():
            raise ValueError('Expected finite25x7 actionchunks')
        return result

    def infer(self, observation):
        images = np.stack([letterbox_rgb(observation[key]) for key in CAMERAS])[None]
        state = np.asarray(observation['observation.state'], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError('Expected seven raw motor-side feedback values')
        # Fixed single-task ACT does not condition on observation.get('prompt').
        return {'actions': self.predict_batch(images, state[None])[0]}

    def get_server_metadata(self):
        return dict(self.metadata)

    def reset(self):
        """Raw action-chunk inference has no episode state or internal queue."""
        return None
