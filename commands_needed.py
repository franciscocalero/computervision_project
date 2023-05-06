!pip install torch
!pip install accelerate
!pip install git+https://github.com/huggingface/diffusers
!git clone https://github.com/franciscocalero/computervision_project.git
!mv -v /content/computervision_project/* /content/
!pip install -r requirements.txt

!accelerate config default
!wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
!wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png

!mkdir "new_model"
!mkdir "validation_images"
!wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
!mv -v v1-5-pruned.ckpt /content/models/

# !accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#  --output_dir="new_models" \
#  --output_validation_dir="validation_images" \
#  --dataset_name=fusing/fill50k \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
#  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --max_train_steps=10 \
#  --validation_steps=2