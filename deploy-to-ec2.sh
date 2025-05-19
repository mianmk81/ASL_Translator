#!/bin/bash
# Deployment script for ASL Translation System to AWS EC2

# Exit on error
set -e

echo "===== ASL Translation System Deployment Script ====="
echo "This script will help you deploy the application to AWS EC2"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    echo "Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check if AWS is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

echo "Step 1: Creating EC2 key pair (if needed)"
KEY_NAME="asl-translator-key"
KEY_FILE="${KEY_NAME}.pem"

if [ ! -f "$KEY_FILE" ]; then
    echo "Creating new key pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "Key pair created and saved to $KEY_FILE"
else
    echo "Key file already exists: $KEY_FILE"
fi

echo ""
echo "Step 2: Creating security group"
SG_NAME="asl-translator-sg"

# Check if security group exists
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating new security group: $SG_NAME"
    SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "Security group for ASL Translator" --query 'GroupId' --output text)
    
    # Add rules
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8080 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 443 --cidr 0.0.0.0/0
    
    echo "Security group created with ID: $SG_ID"
else
    echo "Using existing security group: $SG_ID"
fi

echo ""
echo "Step 3: Launching EC2 instance"
echo "Launching t2.large instance with Ubuntu 22.04..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type t2.large \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":20,\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ASL-Translator}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched with ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
PUBLIC_DNS=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

echo "Instance is running!"
echo "Public IP: $PUBLIC_IP"
echo "Public DNS: $PUBLIC_DNS"

echo ""
echo "Step 4: Waiting for instance to be ready for SSH..."
# Wait a bit for SSH to be ready
sleep 30

echo ""
echo "Step 5: Setting up the server"
echo "Connecting to the server and setting up Docker..."

ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" << 'EOF'
    # Update package lists
    sudo apt update
    sudo apt upgrade -y

    # Install Docker
    sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt update
    sudo apt install -y docker-ce
    sudo usermod -aG docker ubuntu

    # Install Git
    sudo apt install -y git

    # Create project directory
    mkdir -p ~/ASL-Translator
    
    echo "Server setup completed!"
EOF

echo ""
echo "Step 6: Transferring files to the server"
# Create a temporary directory for backend files
mkdir -p temp_deploy/backend

# Copy backend files
cp -r backend/* temp_deploy/backend/
cp Dockerfile temp_deploy/

# Transfer files to the server
scp -i "$KEY_FILE" -r temp_deploy/* ubuntu@"$PUBLIC_IP":~/ASL-Translator/

# Clean up temporary directory
rm -rf temp_deploy

echo ""
echo "Step 7: Building and running the Docker container"
ssh -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" << 'EOF'
    cd ~/ASL-Translator
    sudo docker build -t asl-translator .
    sudo docker run -d -p 8080:8080 --name asl-app asl-translator
    
    echo "Docker container is running!"
EOF

echo ""
echo "===== Deployment Complete! ====="
echo "Your ASL Translation System is now running at: http://$PUBLIC_IP:8080"
echo ""
echo "WebSocket URL for frontend: ws://$PUBLIC_IP:8080/ws/webcam"
echo ""
echo "To SSH into your instance:"
echo "ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo ""
echo "Remember to update your frontend WebSocket URL to point to your EC2 instance!"
