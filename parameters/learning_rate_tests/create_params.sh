TOT_EPOCH=1000000
for CLR in 1e-5 5e-5 1e-4 5e-4 1e-3
  do
  for GLR in 1e-5 5e-5 1e-4 5e-4 1e-3
    do
    cp param_template.json param_C${CLR}_G${GLR}.json
    sed -i "s/GLR/${GLR}/g" param_C${CLR}_G${GLR}.json
    sed -i "s/CLR/${CLR}/g" param_C${CLR}_G${GLR}.json
    sed -i "s/TOT_EPOCH/${TOT_EPOCH}/g" param_C${CLR}_G${GLR}.json
    done
  done
