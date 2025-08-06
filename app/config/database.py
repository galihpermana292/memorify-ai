from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings


# Create async engine
engine = create_async_engine(settings.database_url, echo=False, future=True)

# Create session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# Dependency for FastAPI routes
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# Initialize database (can create tables or just check connection)
async def init_db():
    try:
        async with engine.begin() as conn:
            # If you want to create tables automatically, uncomment:
            # from app.models import Base
            # await conn.run_sync(Base.metadata.create_all)

            # Just check connection
            await conn.execute("SELECT 1")
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
