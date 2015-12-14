rm -rf ddm_normal
rm -rf ddm_rewire
rm -rf er_normal
rm -rf er_rewire
rm -rf pam_normal
rm -rf pam_rewire
rm -rf geo_normal
rm -rf geo_rewire
nohup python3 generate_feature.py 600 pam &
nohup python3 generate_feature.py 600 geo &
nohup python3 generate_feature.py 600 er &
nohup python3 generate_feature.py 600 ddm &
