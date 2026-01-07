from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: str
    uid: int


class UserResponse(BaseModel):
    id: int
    user_name: str
    email: str

    class Config:
        from_attributes = True
