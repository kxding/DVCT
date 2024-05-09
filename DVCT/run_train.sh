cuda_device="0"
img_dir="samples"
output_img_dir="IMG-0305"
model_path="../../s"
chosen_img=""
chosen_token=""
chosen_sim=""

get_animal_image() {
    case "$1" in
        0)
            chosen_img="base-bird"
            chosen_token="bird"
            chosen_sim="bb"
            ;;
        1)
            chosen_img="base-tun"
            chosen_token="mouse"
            chosen_sim="bt"
            ;;
        2)
            chosen_img="bird-multicolor"
            chosen_token="bird"
            chosen_sim="bm"
            ;;
        3)
            chosen_img="bird-simplecolor"
            chosen_token="bird"
            chosen_sim="bs"
            ;;
        4)
            chosen_img="lion-ce"
            chosen_token="lion"
            chosen_sim="lc"
            ;;
        5)
            chosen_img="lion-dark"
            chosen_token="lion"
            chosen_sim="ld"
            ;;
        6)
            chosen_img="lion-zheng"
            chosen_token="lion"
            chosen_sim="lz"
            ;;
        7)
            chosen_img="van"
            chosen_token="van-gogh"
            chosen_sim="van"
            ;;
        8)
            chosen_img="mount"
            chosen_token="mount"
            chosen_sim="mnt"
            ;;
        9)
            chosen_img="sunset"
            chosen_token="sunset"
            chosen_sim="ss"
            ;;
        10)
            chosen_img="beach"
            chosen_token="beach"
            chosen_sim="bh"
            ;;
        11)
            chosen_img="dog"
            chosen_token="dog"
            chosen_sim="dg"
            ;;
        12)
            chosen_img="city"
            chosen_token="city"
            chosen_sim="cy"
            ;;
        13)
            chosen_img="rabbit"
            chosen_token="rabbit"
            chosen_sim="rbt"
            ;;
        14)
            chosen_img="girl2"
            chosen_token="girl"
            chosen_sim="gl2"
            ;;
        15)
            chosen_img="girl3"
            chosen_token="girl"
            chosen_sim="gl3"
            ;;
        16)
            chosen_img="girl4"
            chosen_token="girl"
            chosen_sim="gl4"
            ;;
        17)
            chosen_img="girl5"
            chosen_token="girl"
            chosen_sim="gl5"
            ;;
        18)
            chosen_img="apple"
            chosen_token="apple"
            chosen_sim="ap"
            ;;
        19)
            chosen_img="oringe"
            chosen_token="oringe"
            chosen_sim="og"
            ;;
        20)
            chosen_img="new-bird"
            chosen_token="bird"
            chosen_sim="nb"
            ;;
        21)
            chosen_img="wifi1"
            chosen_token="wifi"
            chosen_sim="wf"
            ;;
        22)
            chosen_img="zebra"
            chosen_token="zebra"
            chosen_sim="zbr"
            ;;
        23)
            chosen_img="horse"
            chosen_token="horse"
            chosen_sim="hrs"
            ;;
        24)
            chosen_img="cat"
            chosen_token="cat"
            chosen_sim="cat"
            ;;

        25)
            chosen_img="dog"
            chosen_token="a white dog with a black big round nose"
            chosen_sim="dog"
            ;;
        26) 
            chosen_img="dog_hat"
            chosen_token="a white dog with a black bid round nose wearing a red hat"
            chosen_sim="dog_hat"
            ;;
        
        27)
            chosen_img="cat_hat"
            chosen_token="a ice tea cat wearing a red hat with left hand up"
            chosen_sim="cat_hat"
            ;;
        28)
            chosen_img="giraff"
            chosen_token="a giraff looking left"
            chosen_sim="giraff"
            ;;
        29) 
            chosen_img="giraff_hat"
            chosen_token="a giraff looking left and wearing a red hat"
            chosen_sim="giraff_hat"
            ;;
        30)
            chosen_img="real-cat"
            chosen_token="a ice tea cat sitting on a chair "
            chosen_sim="real-cat"
            ;;
        31)
            chosen_img="real-cat-mouse"
            chosen_token="a ice tea cat sitting on a chair with its mouth open"
            chosen_sim="real-cat-mouse"
            ;;
        *)
            echo "Invalid index!"
            ;;
    esac
}

get_human_image() {
    case "$1" in
        0)
            chosen_img="base-human"
            chosen_token="human"
            chosen_sim="hb"
            ;;
        1)
            chosen_img="human-complexback"
            chosen_token="human"
            chosen_sim="hc"
            ;;
        2)
            chosen_img="human-simpleback"
            chosen_token="human"
            chosen_sim="hs"
            ;;
        3)
            chosen_img="cartoon-girl"
            chosen_token="cartoon girl"
            chosen_sim="cgl"
            ;;
        4)
            chosen_img="girl"
            chosen_token="girl"
            chosen_sim="gl"
            ;;
        *)
            echo "Invalid index!"
            ;;
    esac
}

train() {
    CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/$1" \
                    --initializer_token="$2" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_dir="output/$3" \
                    --max_train_steps=1000 \
                    --only_save_embeds \
                    --use_l1 --only_train
}

# train2() {
#     for steps in 50 100 150 200 250 300 350 400 450 500
#     do
#         echo "${steps}"
#         CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
#                         --concept_image_dir="./samples/$1" \
#                         --content_image_dir="./samples/$2"  \
#                         --initializer_token="$3" \
#                         --pretrained_model_name_or_path="${model_path}" \
#                         --guidance_scale_train_src=7.5 \
#                         --guidance_scale_train_ref=7.5 \
#                         --output_dir="output/$4-$5" \
#                         --max_train_steps=$steps \
#                         --only_save_embeds \
#                         --use_l1 --only_train
#     done
# }

for i in 30 31
do
    get_animal_image $i
    tar_img=${chosen_img}
    token=${chosen_token}
    tar_sim=${chosen_sim}
    echo "${tar_img}"
    train ${tar_img} ${token} ${tar_sim}
done

# for i in 16 17
# do
#     for j in 16 17
#     do
#         if [ $i -ne $j ]
#         then
#             get_animal_image $i
#             tar_img=${chosen_img}
#             token=${chosen_token}
#             tar_sim=${chosen_sim}
#             get_animal_image $j
#             src_img=${chosen_img}
#             src_sim=${chosen_sim}
#             echo "${tar_img} ${src_img} ${token}"
#             train ${tar_img} ${src_img} ${token} ${tar_sim} ${src_sim}
#         fi
#     done
# done

# for i in 14 15
# do
#     for j in 14 15
#     do
#         if [ $i -ne $j ]
#         then
#             get_animal_image $i
#             tar_img=${chosen_img}
#             token=${chosen_token}
#             tar_sim=${chosen_sim}
#             get_animal_image $j
#             src_img=${chosen_img}
#             src_sim=${chosen_sim}
#             echo "${tar_img} ${src_img} ${token}"
#             train ${tar_img} ${src_img} ${token} ${tar_sim} ${src_sim}
#         fi
#     done
# done

# for i in 13
# do
#     for j in 1
#     do
#         if [ $i -ne $j ]
#         then
#             get_animal_image $i
#             tar_img=${chosen_img}
#             token=${chosen_token}
#             tar_sim=${chosen_sim}
#             get_animal_image $j
#             src_img=${chosen_img}
#             src_sim=${chosen_sim}
#             echo "${tar_img} ${src_img} ${token}"
#             train ${tar_img} ${src_img} ${token} ${tar_sim} ${src_sim}
#         fi
#     done
# done

# for i in 3
# do
#     for j in 4
#     do
#         if [ $i -ne $j ]
#         then
#             get_human_image $i
#             tar_img=${chosen_img}
#             token=${chosen_token}
#             tar_sim=${chosen_sim}
#             get_human_image $j
#             src_img=${chosen_img}
#             src_sim=${chosen_sim}
#             echo "${tar_img} ${src_img} ${token}"
#             train ${tar_img} ${src_img} ${token} ${tar_sim} ${src_sim}
#         fi
#     done
# done