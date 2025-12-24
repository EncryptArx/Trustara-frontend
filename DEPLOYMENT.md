# Deployment Guide - Trustara DeepFake Detection System

This guide will help you deploy the Trustara DeepFake Detection System frontend to Vercel.

## Prerequisites

Before you begin, make sure you have:
- ✅ A GitHub account
- ✅ A Vercel account (sign up at https://vercel.com)
- ✅ Git installed on your machine
- ✅ Node.js and npm installed

## Step 1: Push Code to GitHub

### 1.1 Configure Git Remote

```bash
# Navigate to your project directory
cd c:\Users\Dell\Desktop\DeepFake-Detection-System

# Check current remote (if any)
git remote -v

# Remove existing remote if needed
git remote remove origin

# Add the new Trustara-frontend repository
git remote add origin https://github.com/EncryptArx/Trustara-frontend.git

# Verify the remote was added
git remote -v
```

### 1.2 Stage and Commit Changes

```bash
# Add all your files
git add .

# Commit with a descriptive message
git commit -m "Initial commit: DeepFake Detection System with frontend setup"

# Push to GitHub (you may need to authenticate)
git push -u origin main
```

**Note:** If the repository already has content, you might need to force push:
```bash
git push -u origin main --force
```

## Step 2: Deploy to Vercel

### 2.1 Sign in to Vercel

1. Go to https://vercel.com
2. Click "Sign Up" or "Log In"
3. Sign in with your GitHub account (recommended)

### 2.2 Import Your Project

1. After logging in, click **"Add New..."** button in the top right
2. Select **"Project"**
3. You'll see your GitHub repositories
4. Find **"Trustara-frontend"** and click **"Import"**

### 2.3 Configure Your Project

On the configuration page:

1. **Project Name**: `trustara-deepfake-detection` (or your preferred name)

2. **Framework Preset**: Select **"Vite"** (it should auto-detect)

3. **Root Directory**: Click "Edit" and set it to **`frontend`**
   - This is crucial since your frontend code is in the `frontend` folder!

4. **Build and Output Settings**:
   - Build Command: `npm run build` (auto-filled)
   - Output Directory: `dist` (auto-filled)
   - Install Command: `npm install` (auto-filled)

5. **Environment Variables** (Optional for now):
   - Click "Add Environment Variable"
   - Name: `VITE_API_URL`
   - Value: Your backend API URL (you can add this later)
   - For now, you can skip this and add it after deployment

6. Click **"Deploy"**

### 2.4 Wait for Deployment

- Vercel will now build and deploy your application
- This usually takes 1-3 minutes
- You'll see a progress log showing the build steps

### 2.5 Access Your Deployed App

Once deployment is complete:
- You'll see a success message with confetti! 🎉
- Vercel will provide you with a URL like: `https://trustara-deepfake-detection.vercel.app`
- Click "Visit" to see your live application

## Step 3: Configure Custom Domain (Optional)

If you want a custom domain:

1. Go to your project settings in Vercel
2. Click "Domains" tab
3. Add your custom domain
4. Follow the DNS configuration instructions

## Step 4: Set Up Environment Variables

To connect your frontend to a backend API:

1. In Vercel dashboard, go to your project
2. Click "Settings" tab
3. Click "Environment Variables"
4. Add:
   - Name: `VITE_API_URL`
   - Value: Your backend API URL (e.g., `https://your-backend-api.com`)
5. Click "Save"
6. **Redeploy** your application for changes to take effect:
   - Go to "Deployments" tab
   - Click the three dots on the latest deployment
   - Click "Redeploy"

## Step 5: Enable Automatic Deployments

Vercel automatically sets up continuous deployment:
- Every push to `main` branch will trigger a new deployment
- Pull requests will get preview deployments
- No additional configuration needed!

## Troubleshooting

### Build Fails

**Error: "Cannot find module"**
- Make sure your `package.json` has all dependencies listed
- Check that `root directory` is set to `frontend`

**Error: "Command failed"**
- Ensure build command is `npm run build`
- Check that your code builds locally: `cd frontend && npm run build`

### Frontend Shows but API Calls Fail

- Check that `VITE_API_URL` environment variable is set correctly
- Verify your backend API is deployed and accessible
- Check browser console for CORS errors

### Page Not Found (404) on Refresh

- This is handled by `vercel.json` rewrites
- Make sure `vercel.json` exists in the `frontend` folder

## Project URLs

After deployment, you'll have:
- **Production URL**: `https://your-project.vercel.app`
- **GitHub Repository**: https://github.com/EncryptArx/Trustara-frontend

## Next Steps

1. ✅ Deploy the backend API (Python/FastAPI) to a service like:
   - Railway.app
   - Render.com
   - Google Cloud Run
   - AWS Lambda

2. ✅ Update `VITE_API_URL` environment variable in Vercel with your backend URL

3. ✅ Test the complete flow: Upload → Detection → Results

4. ✅ Set up monitoring and analytics

## Support

If you encounter issues:
- Check Vercel's build logs in the "Deployments" tab
- Review this deployment guide
- Consult Vercel documentation: https://vercel.com/docs

---

**Congratulations! Your Trustara DeepFake Detection frontend is now live! 🚀**
