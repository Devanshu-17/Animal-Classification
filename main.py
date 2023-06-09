import streamlit as st
from fastai.vision.all import *
from PIL import Image
from huggingface_hub import hf_hub_download

model_path = hf_hub_download('espejelomar/fastai-pet-breeds-classification', filename="model.pkl")
model = load_learner(model_path)

cat_breeds = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']
dog_breeds = ['american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1974&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

def add_sbg_from_url():
    st.markdown(
         f"""
         <style>
         .e1fqkh3o3 {{
             background-image: url("https://images.unsplash.com/photo-1584968153986-3f5fe523b044?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=987&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_sbg_from_url()




def app():
    st.title("Pet Shop")

    file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    #write the logo in the sidebar and add padding to the left
    st.sidebar.image("./assets/logo/logo.png", width=250)
    
    #use markdown to write text in bold and large font size
    st.sidebar.markdown("<p style='font-size: 35px; font-family: Times; font-weight: bold; color: white; padding-left: 50px'>Pet Shop</p>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.header("1. Upload an image of an animal.")
    st.sidebar.header("2. The model will predict the type & breed of the animal.")
    st.sidebar.header("3. The model will also suggest other breeds of the animal.")
    st.sidebar.markdown("---")
    
    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)        
        # Preprocess the image and pass it to the model for prediction
        img = PILImage.create(file)
        pred, pred_idx, probs = model.predict(img)
        
        # Check if the predicted breed is a cat or dog breed
        if pred in cat_breeds:
            animal_type = "Cat"
        elif pred in dog_breeds:
            animal_type = "Dog"
        else:
            animal_type = "UNKNOWN"
        
        # Display the predicted breed and the probability
        st.success(f"The predicted animal is of type: {animal_type}")
        st.info(f"Breed of the {animal_type.lower()} is: {pred}")
        probab = round(probs[pred_idx].item()*100, 2)
        #write probability in markdown format with bold text and large font size in a box with green background
        st.markdown(f"<p style='font-size: 30px; font-family: Times; font-weight: bold; color: red;'>Probability of the {animal_type.lower()} being a {pred} is: <b>{probab}%</b></p>", unsafe_allow_html=True)
        
        # Display remaining images of the dog
        if animal_type == "Dog":
            remaining_dog_breeds = [breed for breed in dog_breeds if breed != pred]
            st.title("You may also like: ")
            col1, col2 = st.columns(2)
            for i, dog_breed in enumerate(remaining_dog_breeds):
                image_path = f"./assets/dog/{dog_breed}.jpg"
                with col1 if i%2 == 0 else col2:
                    st.image(image_path, caption=dog_breed, use_column_width=True)
                    #else if the animal is a cat, display remaining images of the cat
            st.header("Recommended Accessories for your pet: ")
            st.image("./assets/dog/dog_accessories.jpeg", caption="Cat Accessories", use_column_width=True)
            #write the accessories for the dog in markdown format
            st.markdown("DOG KIT INCLUDES - ")
            st.markdown("1.  2 Food bowls")
            st.markdown("2.  1 Dog blanket")
            st.markdown("3.  Dog Leash")
            st.markdown("4.  1 Treat dispensing ball")
            st.markdown("5.  1 Lick pad")
            st.markdown("6.  1 Grooming glove")
            st.markdown("7.  Treat Bag")
            st.markdown("8.  2 Collapsible Travel Bowls")
            st.markdown("9.  1 Puppy training clicker")
            st.markdown("10. 1 Potty doorbells")
            st.markdown("11. 1 Teeth cleaning chew ring")
            st.markdown("12. 2 rope toys")
            st.markdown("13. 6 rolls of waste bags with dispenser.")


        elif animal_type == "Cat":
            remaining_cat_breeds = [breed for breed in cat_breeds if breed != pred]
            st.title("You may also like: ")
            col1, col2 = st.columns(2)
            for i, cat_breed in enumerate(remaining_cat_breeds):
                image_path = f"./assets/cat/{cat_breed}.jpg"
                #resize the image to 300x300
                image = Image.open(image_path)
                image = image.resize((500, 500))
                image.save(image_path)
                with col1 if i%2 == 0 else col2:
                    st.image(image_path, caption=cat_breed, use_column_width=True)
                    #else if the animal is unknown, then write a message saying that the animal is not a cat or dog
            st.header("Recommended Accessories for your pet: ")
            st.image("./assets/cat/cat_accessories.jpeg", caption="Cat Accessories", use_column_width=True)
            #write the accessories for the cat in markdown format
            st.markdown("CAT KIT INCLUDES - ")
            st.markdown("1.  Food and water bowls")
            st.markdown("2.  A collar with ID tags")
            st.markdown("3.  A scratching post or pad")
            st.markdown("4.  A comfortable bed or blanket")
            st.markdown("5.  A litter box and scoop")
            st.markdown("6.  A selection of toys (such as interactive toys, balls, and catnip toys)")
            st.markdown("7.  Grooming tools (such as a brush, nail clippers, and toothbrush)")
            st.markdown("8.  A carrier for trips to the vet or travel")
            st.markdown("9.  Treats and food for training and rewards")
            st.markdown("10. A leash and harness for outdoor adventures (if applicable)")

        else:
            st.error("Sorry, the uploaded image is neither a cat nor a dog!")

if __name__ == '__main__':
    app()
    add_bg_from_url()
    add_sbg_from_url()