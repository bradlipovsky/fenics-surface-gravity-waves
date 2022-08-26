# fenics-surface-gravity-waves
Modal analysis of SGWs in variable water depth.

This code is pretty inefficient because it uses ```A.ident_zeros()```, a quick-and-dirty solution. This could be changed.

Run from within Docker using,
```
docker run --name fenics -it -w /home/fenics -v (pwd):/home/fenics/shared -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable
```
