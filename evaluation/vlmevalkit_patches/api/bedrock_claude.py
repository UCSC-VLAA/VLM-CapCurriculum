"""AWS Bedrock Claude API Wrapper for VLMEvalKit"""

import json
import boto3
from botocore.exceptions import ClientError
from vlmeval.smp import *
from vlmeval.api.base import BaseAPI


class BedrockClaude(BaseAPI):
    """AWS Bedrock Claude API Wrapper for Judge Model
    
    Supported Models:
        - global.anthropic.claude-sonnet-4-5-20250929-v1:0 (Claude 4.5 Sonnet - newest)
        - global.anthropic.claude-haiku-4-5-20251001-v1:0 (Claude 4.5 Haiku - newest)
    
    Authentication Methods:
        1. AWS CLI profile (recommended): aws configure
        2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        3. IAM Role (for EC2 instances)
        4. Explicit credentials in kwargs
    
    Example:
        >>> model = BedrockClaude(
        ...     model='global.anthropic.claude-haiku-4-5-20251001-v1:0',
        ...     region='us-east-1'
        ... )
        >>> result = model.generate('What is 2+2?')
    """
    
    is_api: bool = True
    
    def __init__(self,
                 model: str = 'anthropic.claude-3-5-sonnet-20241022-v2:0',
                 region: str = None,
                 profile_name: str = None,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 bedrock_api_key: str = None,
                 retry: int = 10,
                 timeout: int = 300,
                 wait: int = 3,
                 system_prompt: str = None,
                 verbose: bool = True,
                 temperature: float = 0,
                 max_tokens: int = 2048,
                 **kwargs):
        """Initialize Bedrock Claude client.
        
        Args:
            model (str): Bedrock model ID. Defaults to Claude 3.5 Sonnet v2.
            region (str): AWS region. Defaults to 'us-east-1' or from AWS_REGION env var.
            profile_name (str): AWS CLI profile name. Optional.
            aws_access_key_id (str): AWS access key ID. Optional.
            aws_secret_access_key (str): AWS secret access key. Optional.
            bedrock_api_key (str): Bedrock API key (for proxy services). Optional.
                                   Can also be set via BEDROCK_API_KEY environment variable.
            retry (int): Number of retries. Defaults to 10.
            timeout (int): Request timeout in seconds. Defaults to 300.
            wait (int): Wait time between retries. Defaults to 3.
            system_prompt (str): System prompt for the model. Optional.
            verbose (bool): Whether to print verbose logs. Defaults to True.
            temperature (float): Sampling temperature. Defaults to 0.
            max_tokens (int): Maximum tokens to generate. Defaults to 2048.
            **kwargs: Additional arguments passed to BaseAPI.
        """
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Check for Bedrock API key (for proxy services)
        bedrock_api_key = bedrock_api_key or os.environ.get('BEDROCK_API_KEY', None)
        
        if bedrock_api_key:
            # Using Bedrock proxy service with API key
            # Store the key for use in headers
            self.bedrock_api_key = bedrock_api_key
            self.use_proxy = True
            region = region or 'us-east-1'  # Default region for proxy
            self.region = region
            self.bedrock_runtime = None  # Will be set later for proxy mode
            
            if self.verbose:
                self.logger.info(f'Using Bedrock proxy with API key authentication')
        else:
            # Using standard AWS Bedrock
            self.bedrock_api_key = None
            self.use_proxy = False
            
            # Get AWS region from parameter, environment, or default
            region = region or os.environ.get('AWS_REGION', None) or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            self.region = region
            
            # Initialize boto3 client with appropriate credentials
            session_kwargs = {'region_name': region}
            
            if profile_name:
                session_kwargs['profile_name'] = profile_name
                
            if aws_access_key_id and aws_secret_access_key:
                session_kwargs['aws_access_key_id'] = aws_access_key_id
                session_kwargs['aws_secret_access_key'] = aws_secret_access_key
            
            try:
                session = boto3.Session(**session_kwargs)
                self.bedrock_runtime = session.client(
                    service_name='bedrock-runtime',
                    region_name=region
                )
                
                # Test credentials by listing available models (optional)
                # This helps catch auth issues early
                
            except Exception as e:
                error_msg = (
                    f"Failed to initialize AWS Bedrock client: {e}\n\n"
                    f"Please ensure AWS credentials are configured. You can:\n"
                    f"1. Run 'aws configure' to set up AWS CLI\n"
                    f"2. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables\n"
                    f"3. Use an IAM role (if running on EC2)\n"
                    f"4. Pass credentials explicitly via aws_access_key_id and aws_secret_access_key\n"
                    f"5. Use BEDROCK_API_KEY environment variable for proxy services\n\n"
                    f"Also ensure the model {model} is enabled in AWS Bedrock console."
                )
                raise RuntimeError(error_msg)
        
        super().__init__(
            retry=retry,
            wait=wait,
            verbose=verbose,
            system_prompt=system_prompt,
            **kwargs
        )
        
        if self.verbose:
            self.logger.info(f'Initialized BedrockClaude with model: {self.model} in region: {self.region}')
    
    def prepare_inputs(self, inputs):
        """Prepare input messages in Claude format.
        
        Args:
            inputs: Can be str, list of str, list of dicts with 'type' and 'value',
                   or list of dicts with 'role' and 'content'.
        
        Returns:
            list: Messages in Claude API format.
        """
        messages = []
        
        if isinstance(inputs, str):
            # Simple string input
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": inputs}]
            })
            
        elif isinstance(inputs, list):
            if len(inputs) == 0:
                raise ValueError("Empty input list")
                
            if isinstance(inputs[0], dict):
                if 'role' in inputs[0]:
                    # Already in message format: [{"role": "user", "content": [...]}, ...]
                    messages = inputs
                    
                elif 'type' in inputs[0]:
                    # List of content items: [{"type": "text", "value": "..."}, ...]
                    content = []
                    for item in inputs:
                        if item.get('type') == 'text':
                            content.append({
                                "type": "text",
                                "text": item.get('value', '')
                            })
                        # Future: support images if needed for judge model
                        elif item.get('type') == 'image':
                            self.logger.warning('Image input detected but not fully supported in judge mode')
                    
                    messages.append({
                        "role": "user",
                        "content": content
                    })
                else:
                    raise ValueError(f"Unknown dict format in inputs: {inputs[0]}")
            else:
                # List of strings
                text = '\n'.join(str(x) for x in inputs)
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": text}]
                })
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        return messages
    
    def generate_inner(self, inputs, **kwargs):
        """Generate response using Bedrock Claude API.
        
        Args:
            inputs: Input messages (see prepare_inputs for format).
            **kwargs: Additional parameters for the API call.
        
        Returns:
            tuple: (ret_code, answer, response)
                - ret_code: 0 for success, -1 for failure
                - answer: Generated text or fail_msg
                - response: Full response dict or error dict
        """
        try:
            messages = self.prepare_inputs(inputs)
            
            # Get temperature from kwargs or use default, and clamp to [0, 1] for Bedrock
            temperature = kwargs.pop('temperature', self.temperature)
            original_temperature = temperature
            # AWS Bedrock requires temperature to be between 0 and 1
            temperature = max(0.0, min(1.0, float(temperature)))
            
            # Warn if temperature was clamped
            if self.verbose and original_temperature != temperature:
                self.logger.warning(
                    f'Temperature {original_temperature} is outside AWS Bedrock range [0, 1], '
                    f'clamped to {temperature}'
                )
            
            # Build request body for Claude API
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "messages": messages,
                "temperature": temperature,
            }
            
            # Add system prompt if provided
            if self.system_prompt:
                body["system"] = self.system_prompt
            
            # Merge additional kwargs (like top_p, top_k, etc.)
            # Note: top_p also needs to be clamped to [0, 1] if provided
            if 'top_p' in kwargs:
                kwargs['top_p'] = max(0.0, min(1.0, float(kwargs['top_p'])))
            body.update(kwargs)
            
            if self.verbose:
                self.logger.debug(f'Calling Bedrock API with model: {self.model}')
            
            if self.use_proxy:
                # Using proxy service with API key
                import requests
                
                # Proxy endpoint (adjust based on your proxy service)
                proxy_url = os.environ.get('BEDROCK_PROXY_URL', 'https://api.bedrock-proxy.com/v1/invoke')
                
                headers = {
                    'Authorization': f'Bearer {self.bedrock_api_key}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                proxy_body = {
                    'modelId': self.model,
                    'body': body
                }
                
                response = requests.post(
                    proxy_url,
                    headers=headers,
                    json=proxy_body,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    raise Exception(f"Proxy API error: {response.status_code} - {response.text}")
                
                response_body = response.json()
            else:
                # Using standard AWS Bedrock
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model,
                    body=json.dumps(body),
                    contentType='application/json',
                    accept='application/json'
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
            
            # Extract answer from response
            answer = ""
            if 'content' in response_body:
                for content in response_body['content']:
                    if content.get('type') == 'text':
                        answer += content['text']
            
            if not answer:
                self.logger.warning('Empty response from Bedrock API')
                return -1, self.fail_msg, response_body
            
            if self.verbose:
                self.logger.debug(f'Bedrock API response: {answer[:100]}...')
            
            return 0, answer.strip(), response_body
            
        except ClientError as e:
            # AWS Bedrock specific errors
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if self.verbose:
                self.logger.error(f"Bedrock API Error [{error_code}]: {error_message}")
                
                # Provide helpful error messages
                if error_code == 'AccessDeniedException':
                    self.logger.error(
                        "Access denied. Please check:\n"
                        "1. IAM permissions include 'bedrock:InvokeModel'\n"
                        "2. The model is enabled in Bedrock console\n"
                        "3. You're using the correct AWS region"
                    )
                elif error_code == 'ResourceNotFoundException':
                    self.logger.error(
                        f"Model {self.model} not found. Available Claude models:\n"
                        "- anthropic.claude-3-haiku-20240307-v1:0\n"
                        "- anthropic.claude-3-sonnet-20240229-v1:0\n"
                        "- anthropic.claude-3-opus-20240229-v1:0\n"
                        "- anthropic.claude-3-5-sonnet-20240620-v1:0\n"
                        "- anthropic.claude-3-5-sonnet-20241022-v2:0"
                    )
                elif error_code == 'ThrottlingException':
                    self.logger.error("Request throttled. Consider reducing API call rate or requesting higher quota.")
            
            return -1, self.fail_msg, {'error': error_message, 'code': error_code}
            
        except Exception as e:
            # Generic errors
            if self.verbose:
                self.logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
            
            return -1, self.fail_msg, {'error': str(e)}
    
    def working(self):
        """Test if the Bedrock API is working.
        
        Returns:
            bool: True if API is working, False otherwise.
        """
        if self.verbose:
            self.logger.info('Testing Bedrock API connection...')
        
        test_result = self.generate('Hello')
        
        if test_result and test_result != '' and self.fail_msg not in test_result:
            if self.verbose:
                self.logger.info('✅ Bedrock API is working!')
            return True
        else:
            if self.verbose:
                self.logger.error('❌ Bedrock API test failed!')
            return False

