from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from src.models import get_db, User
from src.schemas import Token, TokenData, UserResponse
from src.security import verify_password, create_access_token, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from src.config import settings

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        uid = payload.get("uid")
        if not isinstance(email, str) or not isinstance(uid, int):
            raise credentials_exception
        token_data = TokenData(email=email, uid=uid)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == token_data.uid).first()
    if user is None:
        raise credentials_exception

    is_active: bool = user.is_active  # type: ignore[assignment]
    if not is_active:
        raise credentials_exception
    return user

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username.lower()).first()

    # Timing attack mitigation: always verify password even if user doesn't exist
    if user is None:
        # Verify against dummy hash to maintain constant time
        verify_password(form_data.password, "$2b$12$dummyhashfortimingatttackprotection")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_password: str = user.password  # type: ignore[assignment]
    if not verify_password(form_data.password, user_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    is_active: bool = user.is_active  # type: ignore[assignment]
    if not is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "uid": user.id},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)]
):
    return current_user
