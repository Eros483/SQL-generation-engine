# Agentic SQL generator for Caliper

## Instructions for using the repository
```
git clone https://github.com/Eros483/Caliper-SQL-generator.git
cd Caliper-SQL-generator
cp .env.example .env
```
- Set up env file accordingly.
### Instructions on loading data from sql dump
- Download sql dump data from relevant location.
- The guide assumes the SQL file is named `Dump20250910.sql` and placed in the `data/raw` directory.
```
cd data/raw
sudo apt update
sudo apt install mysql-server mysql-client
sudo systemctl start mysql
sudo mysql_secure_installation
sudo mysql
CREATE USER 'host_name'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON *.* TO 'host_name'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
mysql -u host_name -p < Dump20250910.sql
mysql -u host_name -p -N -e \
"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='fhs_coredb_local';"

```

### Instructions on loading backend
```
pip install -r requirements.txt
python -m backend.main
```

### Instructions on loading frontend
```
cd frontend
npm install
npm run build
```