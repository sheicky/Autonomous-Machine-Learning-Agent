"""
AWS Lambda deployment script for AutoML model
Packages model and creates Lambda function with API Gateway
"""

import boto3
import zipfile
import os
import json
from pathlib import Path

def create_lambda_package(model_path, output_zip='lambda_function.zip'):
    """Create deployment package for Lambda."""
    
    # Lambda handler code
    lambda_handler = '''
import json
import joblib
import pandas as pd
import numpy as np

# Load model at cold start
model = joblib.load('autonomous_model.pkl')

def lambda_handler(event, context):
    """
    Lambda handler for model predictions.
    
    Expected event format:
    {
        "body": {
            "data": [{"feature1": value1, "feature2": value2}]
        }
    }
    """
    try:
        # Parse input
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        data = body.get('data', [])
        
        if not data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No data provided'})
            }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df).tolist()
        
        # Return response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'predictions': predictions.tolist(),
                'probabilities': probabilities,
                'count': len(predictions)
            }, cls=NumpyEncoder)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
'''
    
    # Create zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add handler
        zipf.writestr('lambda_function.py', lambda_handler)
        
        # Add model
        if os.path.exists(model_path):
            zipf.write(model_path, 'autonomous_model.pkl')
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"✅ Lambda package created: {output_zip}")
    return output_zip


def deploy_to_lambda(zip_path, function_name='automl-model-predictor', 
                     role_arn=None, region='us-east-1'):
    """Deploy package to AWS Lambda."""
    
    lambda_client = boto3.client('lambda', region_name=region)
    
    # Read zip file
    with open(zip_path, 'rb') as f:
        zip_content = f.read()
    
    try:
        # Try to update existing function
        response = lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_content
        )
        print(f"✅ Updated existing Lambda function: {function_name}")
        
    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new function
        if not role_arn:
            raise ValueError("role_arn required for new function creation")
        
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_content},
            Timeout=30,
            MemorySize=512,
            Environment={
                'Variables': {
                    'MODEL_VERSION': '1.0.0'
                }
            }
        )
        print(f"✅ Created new Lambda function: {function_name}")
    
    return response


def create_api_gateway(lambda_arn, api_name='AutoML-Model-API', region='us-east-1'):
    """Create API Gateway for Lambda function."""
    
    apigateway = boto3.client('apigateway', region_name=region)
    lambda_client = boto3.client('lambda', region_name=region)
    
    # Create REST API
    api_response = apigateway.create_rest_api(
        name=api_name,
        description='AutoML Model Prediction API',
        endpointConfiguration={'types': ['REGIONAL']}
    )
    
    api_id = api_response['id']
    
    # Get root resource
    resources = apigateway.get_resources(restApiId=api_id)
    root_id = resources['items'][0]['id']
    
    # Create /predict resource
    resource = apigateway.create_resource(
        restApiId=api_id,
        parentId=root_id,
        pathPart='predict'
    )
    
    resource_id = resource['id']
    
    # Create POST method
    apigateway.put_method(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        authorizationType='NONE'
    )
    
    # Set up Lambda integration
    apigateway.put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri=f'arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
    )
    
    # Deploy API
    deployment = apigateway.create_deployment(
        restApiId=api_id,
        stageName='prod'
    )
    
    # Add Lambda permission for API Gateway
    lambda_client.add_permission(
        FunctionName=lambda_arn.split(':')[-1],
        StatementId='apigateway-invoke',
        Action='lambda:InvokeFunction',
        Principal='apigateway.amazonaws.com',
        SourceArn=f'arn:aws:execute-api:{region}:*:{api_id}/*/*'
    )
    
    api_url = f'https://{api_id}.execute-api.{region}.amazonaws.com/prod/predict'
    print(f"✅ API Gateway created: {api_url}")
    
    return api_url


def main():
    """Main deployment script."""
    
    print("🚀 AWS Lambda Deployment for AutoML Model\n")
    
    # Configuration
    MODEL_PATH = 'autonomous_model.pkl'
    FUNCTION_NAME = 'automl-model-predictor'
    ROLE_ARN = os.getenv('AWS_LAMBDA_ROLE_ARN')  # Set this in environment
    REGION = os.getenv('AWS_REGION', 'us-east-1')
    
    # Step 1: Create package
    print("Step 1: Creating Lambda package...")
    zip_path = create_lambda_package(MODEL_PATH)
    
    # Step 2: Deploy to Lambda
    print("\nStep 2: Deploying to Lambda...")
    lambda_response = deploy_to_lambda(zip_path, FUNCTION_NAME, ROLE_ARN, REGION)
    lambda_arn = lambda_response['FunctionArn']
    
    # Step 3: Create API Gateway
    print("\nStep 3: Creating API Gateway...")
    api_url = create_api_gateway(lambda_arn, region=REGION)
    
    print("\n✅ Deployment Complete!")
    print(f"\n📍 API Endpoint: {api_url}")
    print("\n📝 Test with:")
    print(f'''
curl -X POST {api_url} \\
  -H "Content-Type: application/json" \\
  -d '{{"data": [{{"feature1": 1.0, "feature2": 2.0}}]}}'
''')


if __name__ == '__main__':
    main()
