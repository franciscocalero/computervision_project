# !git clone https://github.com/franciscocalero/computervision_project.git
# !pip install torch
# !pip install accelerate
# !pip install git+https://github.com/huggingface/diffusers
# !mv -v /content/computervision_project/* /content/
# !pip install -r requirements.txt
# !accelerate config default
# !mkdir "final_models"
# !mkdir "validation_images"

######### Fill 50K Sample #########
# !accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#  --output_dir="final_models" \
#  --output_validation_dir="validation_images" \
#  --dataset_name=fusing/fill50k \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --train_batch_size=4 \
#  --gradient_accumulation_steps=4 \
#  --max_train_steps=10000 \
#  --validation_steps=500 \
#  --checkpointing_steps=500 \
#  --image_column='image' \
#  --conditioning_image_column='conditioning_image' \
#  --caption_column='text'

######### Dog Poses Simple Sample #########
!accelerate launch train_controlnet.py  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir="final_models"  --output_validation_dir="validation_images"  --dataset_name="JFoz/dog-poses-controlnet-dataset"  --resolution=512  --learning_rate=1e-5  --train_batch_size=4  --gradient_accumulation_steps=4  --max_train_steps=10  --validation_steps=10  --checkpointing_steps=10  --image_column='original_image'  --conditioning_image_column='conditioning_image'  --caption_column='caption'

!accelerate launch train_controlnet.py  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir="final_models"  --output_validation_dir="validation_images"  --dataset_name="JFoz/dog-poses-controlnet-dataset"  --resolution=128  --learning_rate=1e-5  --train_batch_size=1  --gradient_accumulation_steps=1  --max_train_steps=10  --validation_steps=10  --checkpointing_steps=10  --image_column='original_image'  --conditioning_image_column='conditioning_image'  --caption_column='caption'

######### Dog Poses Simple One Epoch #########
!accelerate launch train_controlnet.py  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir="final_models"  --output_validation_dir="validation_images"  --dataset_name="JFoz/dog-poses-controlnet-dataset"  --resolution=512  --learning_rate=1e-5  --train_batch_size=4  --gradient_accumulation_steps=4  --num_train_epochs=1  --validation_steps=500  --checkpointing_steps=500  --image_column='image'  --conditioning_image_column='conditioning_image'  --caption_column='text'
