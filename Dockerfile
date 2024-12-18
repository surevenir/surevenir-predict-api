# Gunakan image dasar Python
FROM python:3.11-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt ./

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh aplikasi ke dalam container
COPY . .

# Tentukan port yang digunakan oleh aplikasi
EXPOSE 5000

# Tentukan command untuk menjalankan aplikasi
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
