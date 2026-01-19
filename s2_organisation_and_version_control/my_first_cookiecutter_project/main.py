from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
import re


from contextlib import asynccontextmanager
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import cv2





class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Hello")
    yield
    print("Goodbye")

app = FastAPI()

# GET requests
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: ModelName):
    return {"item_id": item_id}

@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}



@app.get("/text_model/")
def get_email_type(data: str):
    regex = r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)\.[A-Z|a-z]{2,}\b'
    match = re.fullmatch(regex, data)
    email_type = None
    if match:
        domain = match.group(1)
        if '.' in domain:
            # Take main domain part if subdomain exists
            main_domain = domain.split('.')[-2]
            email_type = main_domain
        else:
            email_type = domain
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": match is not None,
        "type": email_type
    }
    return response

# POST
database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
    username_db = database['username']
    password_db = database['password']
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)

        img = cv2.imread("image.jpg")
        height, width = 224, 224
        res = cv2.resize(img, (width, height))
        cv2.imwrite('image_resize.jpg', res)
        image.close()

    return FileResponse('image_resize.jpg')