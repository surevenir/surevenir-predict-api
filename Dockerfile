# Gunakan image dasar Python
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt ./

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh aplikasi ke dalam container
COPY . .

# Tentukan port yang digunakan oleh aplikasi
EXPOSE 8080

# Tentukan command untuk menjalankan aplikasi
CMD ["python", "app.py"]
