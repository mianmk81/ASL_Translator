# ASL Translation System Deployment Guide

This guide explains how to deploy the ASL Translation System to AWS EC2.

## Prerequisites

- AWS account
- AWS CLI installed and configured (optional, but recommended)
- Git (optional)

## Deployment Options

### Option 1: Using the Deployment Script (Recommended)

1. Make sure you have AWS CLI installed and configured with your credentials:
   ```
   aws configure
   ```

2. Make the deployment script executable:
   ```
   chmod +x deploy-to-ec2.sh
   ```

3. Run the deployment script:
   ```
   ./deploy-to-ec2.sh
   ```

4. The script will:
   - Create a key pair for SSH access
   - Create a security group with necessary ports open
   - Launch a t2.large EC2 instance
   - Install Docker on the instance
   - Transfer your application files
   - Build and run the Docker container

5. After deployment, update your frontend WebSocket URL to point to your EC2 instance:
   ```javascript
   // In SignToText.js
   const ws = new WebSocket('ws://your-ec2-public-ip:8080/ws/webcam');
   ```

### Option 2: Manual Deployment

Follow these steps to manually deploy your application:

#### 1. Launch an EC2 Instance

1. Sign in to AWS Console
2. Navigate to EC2 Dashboard
3. Click "Launch Instance"
4. Choose "Ubuntu Server 22.04 LTS"
5. Select t2.large or better
6. Create a new key pair and download it
7. Configure security group to allow ports 22, 80, 8080, and 443
8. Launch the instance

#### 2. Connect to Your Instance

```bash
chmod 400 your-key-pair.pem
ssh -i your-key-pair.pem ubuntu@your-instance-public-ip
```

#### 3. Install Docker

```bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce
sudo usermod -aG docker ubuntu
newgrp docker
```

#### 4. Transfer Files

From your local machine:

```bash
scp -i your-key-pair.pem -r /path/to/ASL-Copy/* ubuntu@your-instance-public-ip:~/ASL-Translator/
```

#### 5. Build and Run Docker Container

```bash
cd ~/ASL-Translator
docker build -t asl-translator .
docker run -d -p 8080:8080 --name asl-app asl-translator
```

## Updating Your Application

To update your application after making changes:

```bash
# Transfer new files to the server
scp -i your-key-pair.pem -r /path/to/updated/files/* ubuntu@your-instance-public-ip:~/ASL-Translator/

# SSH into the server
ssh -i your-key-pair.pem ubuntu@your-instance-public-ip

# Rebuild and restart the container
cd ~/ASL-Translator
docker build -t asl-translator .
docker stop asl-app
docker rm asl-app
docker run -d -p 8080:8080 --name asl-app asl-translator
```

## Troubleshooting

### Check if the container is running:
```bash
docker ps
```

### View container logs:
```bash
docker logs asl-app
```

### Check if the application is accessible:
```bash
curl http://localhost:8080
```

### If the WebSocket connection fails:
- Ensure port 8080 is open in your security group
- Check that your frontend is using the correct WebSocket URL
- Verify the application is running with `docker ps`
- Check logs for errors with `docker logs asl-app`
