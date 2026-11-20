set -eu
cd /data/b601-recap-round1/evo
(
 curl -fsSL --retry 3 --connect-timeout 10 https://codeload.github.com/MINT-SJTU/Evo-RL/tar.gz/b1009248c270515946b96ca90ae5a156f103c637 -o sources/evorl.tar.gz
 tar -xzf sources/evorl.tar.gz --strip-components=1 -C sources/Evo-RL
 printf '%s\n' b1009248c270515946b96ca90ae5a156f103c637 > sources/Evo-RL.commit
) > sources/download-source.log 2>&1 &
for f in config.json value_train_config.json pistar06_save_info.json policy_preprocessor.json policy_preprocessor_step_1_normalizer_processor.safetensors model.safetensors; do
 (
 base=https://hf-mirror.com/BetterCallSaul04/evorl_pistar06_value_stacking_r1/resolve/cc9186cd31d86b357a18ca3e0beedd8d88d222f4
 dest=inputs/stacking/$f
 if [ ! -s "$dest" ]; then
   curl -fLsS --retry 3 --connect-timeout 10 --max-time 1400 "$base/$f" -o "$dest.partial"
   mv "$dest.partial" "$dest"
 fi
 ) > "sources/download-$f.log" 2>&1 &
done
wait
du -h inputs/stacking/* | tail
