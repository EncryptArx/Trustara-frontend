import React, {useState, useEffect} from 'react'
import Uploader from './components/Uploader'
import ResultPanel, { DeepfakeResult } from './components/ResultPanel'

export default function App(){
  const [result, setResult] = useState<DeepfakeResult | null>(null)

  // Generate floating orbs
  useEffect(() => {
    const orbContainer = document.createElement('div')
    orbContainer.className = 'orb-container'
    
    // Create 3 large floating orbs
    const orb1 = document.createElement('div')
    orb1.className = 'orb orb-1'
    
    const orb2 = document.createElement('div')
    orb2.className = 'orb orb-2'
    
    const orb3 = document.createElement('div')
    orb3.className = 'orb orb-3'
    
    orbContainer.appendChild(orb1)
    orbContainer.appendChild(orb2)
    orbContainer.appendChild(orb3)
    
    document.body.appendChild(orbContainer)

    return () => {
      document.body.removeChild(orbContainer)
    }
  }, [])

  return (
    <div className="min-h-screen flex items-center justify-center p-4 md:p-8 relative z-10">
      <div className="w-full max-w-7xl">
        <header className="mb-12 md:mb-16 text-center animate-fade-in">
          <div className="inline-block mb-4">
            <div className="badge badge-info">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
              </svg>
              Powered by Advanced AI
            </div>
          </div>
          
          <h1 className="text-6xl md:text-8xl font-black mb-4 gradient-text" 
              style={{letterSpacing: '-0.02em'}}>
            DeepSecure
          </h1>
          
          <p className="text-lg md:text-xl text-gray-400 font-light max-w-2xl mx-auto">
            Next-Generation Deepfake Detection Platform
          </p>
          
          <div className="mt-8 flex items-center justify-center gap-6 text-sm text-gray-500">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span>99.8% Accuracy</span>
            </div>
            <div className="w-px h-4 bg-gray-700"></div>
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
              <span>Secure & Private</span>
            </div>
            <div className="w-px h-4 bg-gray-700"></div>
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span>Instant Results</span>
            </div>
          </div>
        </header>

        {result ? (
          <ResultPanel result={result} onRetry={()=>setResult(null)} />
        ) : (
          <Uploader onResult={(r: DeepfakeResult)=>setResult(r)} />
        )}
      </div>
    </div>
  )
}
