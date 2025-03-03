import dill
def hello_world(object):
    print(f"Heelo world by {object}")
def save_object(object):
    with open("data.pkl","wb") as file:
        dill.dump(object,file)
def load_object(file_path):
    with open(file_path,"rb") as file:
        return dill.load(file) 

if __name__ == "__main__":
    name1 = "Jigme"
    save_object(hello_world)
    loaded_func = load_object("data.pkl")
    loaded_func(name1)
    