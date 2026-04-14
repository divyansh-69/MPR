# 🍃Plant-Disease-Detection🍃
* In agriculture, timely disease diagnosis can make the difference between profit and loss. Our project, Plant Disease Detection AI, harnesses the power of Deep Learning to empower farmers with instant disease recognition. Using a CNN model implemented in PyTorch, it analyzes leaf images and accurately classifies them into 39 different disease types. Trained on the comprehensive PlantVillage dataset, this system demonstrates how artificial intelligence can revolutionize sustainable farming practices.
## 🌿Run Project in your Machine
* You must have **Python3.11** installed in your machine.
* Create a Python Virtual Environment & Activate Virtual Environment <br/> 
 python -m venv venv <br/>
 venv\Scripts\activate
* Install all the dependencies using below command
    `pip install -r requirements.txt` <br/>
    Updated Requirements (paste this in requirement.txt) <br/>
click==7.1.2  <br/>
Flask==1.1.2  <br/>
gunicorn==20.1.0  <br/>
itsdangerous==1.1.0  <br/>
Jinja2==2.11.3  <br/>
MarkupSafe==1.1.1  <br/>
numpy>=1.24.4  <br/>
pandas>=1.5.3  <br/>
Pillow>=9.0.0  <br/>
python-dateutil>=2.8.2  <br/>
pytz>=2023.3  <br/>
six>=1.16.0  <br/>
torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html  <br/>
torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html  <br/>
typing-extensions>=4.7.1  <br/>
Werkzeug>=2.2.3  <br/>

* Go to the `Flask Deployed App` folder.
* Download the pre-trained model file `plant_disease_model_1.pt` from [here](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link)
* Add the downloaded file in `Flask Deployed App` folder.
* Run the Flask app using below command `python app.py`
## 🌿Testing Images
* If you don’t have your own leaf images, you can use the sample test images provided in the test_images folder.
* Each image in this folder is labeled with its corresponding disease name, allowing you to easily verify the model’s accuracy and performance.
* Simply upload these images to the model interface or run them through the prediction script to confirm that the system correctly identifies each disease.
## 🖥️ Snippet of the Web Application

<img width="1919" height="913" alt="Screenshot 2025-10-17 005503" src="https://github.com/user-attachments/assets/b3acc097-0edd-4f8a-bcfc-b04a9e85251f" />  <br/>


<img width="1898" height="910" alt="Screenshot 2025-10-17 010009" src="https://github.com/user-attachments/assets/7649e6e0-4f1c-4e1b-b3a5-e19b60bf035f" />  <br/>



<img width="1918" height="911" alt="Screenshot 2025-10-17 005715" src="https://github.com/user-attachments/assets/f2c114e4-e3a8-4df6-8671-7335dd5d173c" />  <br/>




<img width="1919" height="902" alt="Screenshot 2025-10-17 005815" src="https://github.com/user-attachments/assets/26bda287-7de5-4191-a9e8-db36e33900e5" />
