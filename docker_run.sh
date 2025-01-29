docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it -v `pwd`:/project/anymorph:rw --hostname $HOSTNAME --workdir /project/anymorph/modular-rl/src/scripts/ anymorph
