# 🚀 Quick Vercel Deployment Steps

Your code is now on GitHub! Follow these steps to deploy to Vercel:

## Step 1: Go to Vercel
Visit: https://vercel.com and sign in with your GitHub account

## Step 2: Import Project
1. Click **"Add New..."** → **"Project"**
2. Find **"Trustara-frontend"** in your repositories
3. Click **"Import"**

## Step 3: Configure Project Settings
**IMPORTANT:** Set these configurations:

- **Root Directory**: `frontend` ⚠️ (Click Edit and select the frontend folder)
- **Framework**: Vite (should auto-detect)
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

## Step 4: Deploy
Click the **"Deploy"** button and wait 1-3 minutes

## Step 5: Get Your Live URL
Once complete, you'll receive a URL like:
`https://trustara-frontend.vercel.app`

---

## 🔧 After Deployment

### Add Backend API URL (When Ready)
1. Go to your project in Vercel
2. Settings → Environment Variables
3. Add: `VITE_API_URL` = `your-backend-url`
4. Redeploy from Deployments tab

---

## 📋 Important Notes

✅ **Root Directory MUST be `frontend`** - This is the most important setting!

✅ Every push to GitHub will auto-deploy

✅ Full deployment guide available in `DEPLOYMENT.md`

---

Need help? Check the detailed `DEPLOYMENT.md` file in your repository!
