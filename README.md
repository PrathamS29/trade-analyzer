# Trade Analyzer - Full Stack Application

A modern fullstack application built with React, FastAPI, and MongoDB, featuring JWT authentication.

## Tech Stack

### Frontend
- **React** - UI framework
- **Vite** - Build tool and dev server
- **React Router** - Routing
- **Axios** - HTTP client
- **Context API** - State management

### Backend
- **FastAPI** - Python web framework
- **Beanie ODM** - MongoDB ODM with Pydantic integration
- **MongoDB** - NoSQL database with Motor (async driver)
- **JWT** - JSON Web Tokens for authentication
- **Passlib** - Password hashing with bcrypt

## Contributors
Built by **Pratham Subrahmanya and Shaun Gao** - Full-stack development, authentication system, MongoDB integration, and API design
- GitHub: [@PrathamS29](https://github.com/PrathamS29) (personal) | [@PrathamS-23](https://github.com/PrathamS-23) (school)
- Collaborated with [@shaungao123](https://github.com/shaungao123)

## Project Structure

```
trade-analyzer/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── main.py      # FastAPI application entry point
│   │   ├── database.py  # MongoDB connection
│   │   ├── models.py    # Pydantic models
│   │   ├── auth.py      # JWT authentication utilities
│   │   └── routers/     # API routes
│   │       ├── auth.py  # Authentication routes
│   │       └── users.py # User routes
│   ├── requirements.txt # Python dependencies
│   └── README.md        # Backend documentation
│
├── frontend/            # React frontend
│   ├── src/
│   │   ├── pages/       # Page components
│   │   ├── components/  # Reusable components
│   │   ├── contexts/    # React contexts
│   │   └── services/    # API services
│   ├── package.json     # Node dependencies
│   └── README.md        # Frontend documentation
│
└── README.md            # This file

```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB (local or Atlas)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file:
```bash
cp .env.example .env
```

6. Update `.env` with your MongoDB connection string:
```
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=trade_analyzer
SECRET_KEY=your-secret-key-change-this-in-production
```

7. Run the backend server:
```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

API Documentation: http://localhost:8000/docs

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The app will be available at http://localhost:3000

## Features

### Authentication
- User registration
- User login
- JWT-based authentication
- Protected routes
- Automatic token management

### API Endpoints

#### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login and get JWT token

#### Users
- `GET /api/users/me` - Get current user info (requires authentication)
- `GET /api/users/protected` - Protected route example (requires authentication)

## Development

### Running Both Servers

You'll need to run both servers in separate terminals:

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Testing the Application

1. Start MongoDB (if running locally)
2. Start the backend server
3. Start the frontend server
4. Navigate to http://localhost:3000
5. Register a new user
6. Login with your credentials
7. Access the protected dashboard

## Environment Variables

### Backend (.env)
- `MONGODB_URL` - MongoDB connection string
- `DATABASE_NAME` - Database name
- `SECRET_KEY` - Secret key for JWT tokens

## Security Notes

- Change the `SECRET_KEY` in production
- Use environment variables for sensitive data
- Keep dependencies updated
- Implement rate limiting for production
- Use HTTPS in production
- Implement proper CORS configuration for production

## ✅ Status: FULLY WORKING

The application has been tested and is fully functional:

- ✅ **Backend**: FastAPI with Beanie ODM working perfectly
- ✅ **Database**: MongoDB connection and operations successful
- ✅ **Authentication**: JWT tokens and password hashing working
- ✅ **User Management**: Registration, login, and protected routes working
- ✅ **Frontend**: React app builds and runs successfully
- ✅ **Compatibility**: Fixed Python 3.13 compatibility issues

## Quick Start

1. **Setup**: `./setup.sh`
2. **Start Backend**: `cd backend && source venv/bin/activate && uvicorn app.main:app --reload`
3. **Start Frontend**: `cd frontend && npm run dev`
4. **Access**: http://localhost:3000

## License

MIT

