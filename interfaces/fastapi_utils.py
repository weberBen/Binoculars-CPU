import os
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi.security.api_key import APIKeyHeader
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from typing import Union, Optional, List
from config import API_SECRET_KEY, API_ENCRYPT_ALGORITHM, API_ACCESS_TOKEN_EXPIRE_MINUTES, API_AUTHORIZED_API_KEYS

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

# Security schemes
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=True,
    description="Your API key for authentication. Must be one of the authorized keys configured on the server."
)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    description="JWT token for authentication"
)

# Token generation
def create_access_token(api_key: str, expires_delta: Optional[timedelta] = None):
    to_encode = {"key": api_key}
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=API_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, API_SECRET_KEY, algorithm=API_ENCRYPT_ALGORITHM)
    
    return encoded_jwt


# Token validation
async def validate_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, API_SECRET_KEY, algorithms=[API_ENCRYPT_ALGORITHM])
        api_key: str = payload.get("key")
        if api_key is None or (api_key not in API_AUTHORIZED_API_KEYS):
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return api_key
