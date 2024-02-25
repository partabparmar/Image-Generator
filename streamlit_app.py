import streamlit as st
import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the prior model pipeline
pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
pipe_prior.to("cuda")

# Load the text-to-image model pipeline
t2i_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
t2i_pipe.to("cuda")

# Define the Streamlit app
def main():
    st.title("Text-to-Image Generation")

    # User input for prompt
    user_prompt = st.text_input("Enter a prompt:")

    if st.button("Generate Image"):
        # Generate image using the pipelines
        image_embeds, negative_image_embeds = pipe_prior(user_prompt, guidance_scale=1.0).to_tuple()
        generated_image = t2i_pipe(user_prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds).images[0]

        # Save the generated image
        image_path = "generated_image.png"
        generated_image.save(image_path)

        # Display the generated image
        st.image(mpimg.imread(image_path), caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
