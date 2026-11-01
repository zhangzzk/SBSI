#!/bin/bash
#SBATCH -p cluster
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -t 04:00:00
#SBATCH -J t1manifest
#SBATCH -o /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/cleanup/manifest_%j.log
# Tier-1 cleanup approved by the owner 2026-11-01: build the exact deletion list (size path). Deletes nothing.
set -euo pipefail
O=/home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/cleanup
CP=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior
U=/project/ls-gruen/users/zekang.zhang
: > $O/files.txt
for s in galsbi_pair1m_20261012 galsbi_blind100k_20261003 galsbi_mock500k_20261006 galsbi_faintg05_20261013 \
         galsbi_blind10k_20261003 galsbi_twin100k_20261005 galsbi_constgold100k_20261025 optimize_20261011 \
         galsbi_mock10k_20261002 galsbi_throughput_20261002 galsbi_bright_mh_20261002 galsbi_speed_20261003 \
         galsbi_speedup_20261003 galsbi_concurrency_20261026_v2 galsbi_mh1024_20261002 galsbi_bias238_20261002 \
         galsbi_long_mh_20261001 galsbi_light_transfer_20261002 galsbi_cpu_scaling_20261002 galsbi_tempering_20261002 \
         galsbi_concurrency_20261026_v1; do
  find "$CP/$s" -type f \( -name chain.npz -o -name local_profiles.npz -o -name final_scenes.npz \) -printf '%s\t%p\n' >> $O/files.txt
done
find $CP/galsbi_concurrency_20261026_v1 $CP/galsbi_concurrency_20261026_v2 -type d -name __pycache__ -prune -printf '%p\n' > $O/pycache_dirs.txt
ls -d $U/lsst_sims_fs2_25876_anchorblend_eachpair_g?_rank??_pilot_c400-409 $U/lsst_sims_fs2_25876_anchorblend_coherent_g2_pilot_c400-409 \
      $U/lsst_sims_fs2_25876_anchorblend_smoke $U/lsst_sims_fs2_25876_v22cgq3_c48-139_*_failed1563???? \
      $U/lsst_sims_fs2_25876_v22decomp_rankpilot_rank* $U/lsst_sims_fs2_25876_v22decomp_pilot_* $U/lsst_sims_fs2_25876_v22cgq3_pilot_* \
      $U/lsst_sims_fs2_25876_constant_REPRO > $O/stale_dirs.txt
ls -d $U/sbsi_catalogues/*.oldcats_bak > $O/bak_files.txt
echo "files: $(wc -l < $O/files.txt)"
awk -F'\t' '{s+=$1} END {printf "apparent bytes %.1f GB\n", s/1e9}' $O/files.txt
cut -f2 $O/files.txt | awk -F/ '{print $8}' | sort | uniq -c
echo "pycache dirs: $(wc -l < $O/pycache_dirs.txt)"
cat $O/pycache_dirs.txt | xargs -d '\n' -I{} find {} -type f | wc -l
echo "stale dirs: $(wc -l < $O/stale_dirs.txt)"
cat $O/stale_dirs.txt | xargs -d '\n' du -sc --block-size=1M | tail -1
cat $O/bak_files.txt
echo MANIFEST_DONE
