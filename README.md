# Installation
## Add trained model
Before use the app we must add a trained model. We can train one in a python enviroment with Jupyterlab. After that, we add the model labeled <code>model.keras</code> to the <code>ai-service</code> folder in the root app after cloned the repository.

Read the <code>BUILD_MODEL.md</code> file.

## Run the app
##### 1.- First we clone the repo
```
git clone git@github.com:u83mm/cat_vs_dogs_IA.git
```
##### 2.- Navigate to the "cat_vs_dogs_IA" and add the trained model to the "ai-service" folder
```
cd cat_vs_dogs_IA
./ai-service/
       |_ model.keras
```

##### 3.- Add the .env file with database credentials to "Application/MariaDB" folder:
```
MYSQL_ROOT_PASSWORD=admin
MYSQL_DATABASE=my_database
MYSQL_USER=admin
MYSQL_PASSWORD=admin
```
##### 4.- Run:
```
docker compose build
docker compose up -d
```

##### 5.- Install dependencies:
```
docker exec -it php bash
composer install
exit
```

If all is OK, now you can enter in URL:
http://localhost


```
Go to the "AI-dogs" menu, upload a cat or dog image and click "Test Image" button.
```
