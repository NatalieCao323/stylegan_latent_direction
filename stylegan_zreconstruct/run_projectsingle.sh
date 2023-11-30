mkdir results/w
for 2 3 4 5 6 11
do
    python3 projector.py --ckpt checkpoints/stylegan-1024px-new.model --files single_reconstruct/images/""$i --step 1000 --results single_reconstruct/projects1024/10000 --noise_regularize 10000 --size 1024 > log""$i""_10000.txt
done
