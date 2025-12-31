# Installation
## Add trained model
Before use the app we must add a trained model. We can train one in a python enviroment with Jupyterlab. After that, we add the model labeled "perros_gatos_master.keras" to the "ai-service" folder in the root app after cloned the repository.

## Run the app
##### 1.- First we clone the repo
```
git clone git@github.com:u83mm/cat_vs_dogs_IA.git
```
##### 2.- Navigate to the "cat_vs_dogs_IA" and add the trained model to the "ai-service" folder
```
cd cat_vs_dogs_IA
./ai-service/
       |_ perros_gatos_master.keras
```

##### 3.- Add the .env file with database credentials to "Application/MariaDB" folder:
```
MYSQL_ROOT_PASSWORD=admin
MYSQL_DATABASE=my_database
MYSQL_USER=admin
MYSQL_PASSWORD=admin
```
##### 4.- Add necessary folders:
```
cd Application
mkdir db_vol log log/php log/apache log/db
```
##### 5.- Run:
```
docker compose build
docker compose up -d
```

##### 6.- Install dependencies:
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
