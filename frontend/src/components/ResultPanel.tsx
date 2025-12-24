import React, {useEffect, useRef, useState} from 'react'
import { DeepfakeResult } from './types'

export default function ResultPanel({ result, onRetry }: { result: DeepfakeResult, onRetry: ()=>void }){
  const progressRef = useRef<HTMLDivElement|null>(null)
  const [showToast, setShowToast] = useState(false)
  const [toastMessage, setToastMessage] = useState('')

  useEffect(()=>{
    if(progressRef.current){
      const v = Math.max(0, Math.min(100, result.confidence))
      progressRef.current.style.width = '0%'
      requestAnimationFrame(()=>{
        setTimeout(() => {
          if (progressRef.current) {
            progressRef.current.style.transition = 'width 1400ms cubic-bezier(0.16, 1, 0.3, 1)'
            progressRef.current.style.width = `${v}%`
          }
        }, 100)
      })
    }
  }, [result])

  const isDeepfake = result.label === 'DEEPFAKE'
  const isReal = result.label === 'REAL'
  const labelColor = isDeepfake ? '#ef4444' : isReal ? '#10b981' : '#00d4ff'
  const badgeClass = isDeepfake ? 'badge-danger' : isReal ? 'badge-success' : 'badge-info'

  function showNotification(message: string) {
    setToastMessage(message)
    setShowToast(true)
    setTimeout(() => setShowToast(false), 2000)
  }

  function copyExact(){
    const text = `Result\nLabel: ${result.label}\nConfidence: ${result.confidence.toFixed(2)}%\nMedia type: ${result.media_type}\nTimestamp: ${result.timestamp}\nModel: ${result.model}\nGeo-tag: ${result.geo_tag}\n\nPreview Photo\n`
    navigator.clipboard.writeText(text).then(()=> showNotification('✓ Copied to clipboard!'))
  }

  function downloadReport(){
    const a = document.createElement('a')
    const data = JSON.stringify(result, null, 2)
    a.href = URL.createObjectURL(new Blob([data], { type: 'application/json' }))
    a.download = `deepsecure_report_${Date.now()}.json`
    a.click()
    showNotification('✓ Report downloaded!')
  }

  function share(){
    const payload = `DeepSecure Analysis\nLabel: ${result.label}\nConfidence: ${result.confidence.toFixed(2)}%\nModel: ${result.model}`
    if(navigator.share){
      navigator.share({ title: 'DeepSecure Result', text: payload })
        .then(() => showNotification('✓ Shared successfully!'))
        .catch(()=> {
          navigator.clipboard.writeText(payload)
          showNotification('✓ Copied to clipboard!')
        })
    } else {
      navigator.clipboard.writeText(payload)
      showNotification('✓ Copied to clipboard!')
    }
  }

  return (
    <div className="glass-card p-8 md:p-12 rounded-3xl animate-scale-in relative overflow-hidden">
      <div className="relative z-10">
        <div className="flex flex-col lg:flex-row gap-10">
          {/* Main Result Panel */}
          <div className="flex-1 space-y-8">
            {/* Status Badge */}
            <div className={`badge ${badgeClass} inline-flex`}>
              <div 
                className="w-2 h-2 rounded-full animate-pulse"
                style={{backgroundColor: labelColor}}
              />
              Analysis Complete
            </div>

            {/* Main Result */}
            <div className="gradient-border p-1">
              <div className="glass-card p-8 rounded-2xl">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <div className="text-sm text-gray-400 mb-2 uppercase tracking-wider">Detection Result</div>
                    <div 
                      className="text-5xl md:text-6xl font-black uppercase"
                      style={{color: labelColor}}
                    >
                      {result.label}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-400 mb-2">Confidence</div>
                    <div 
                      className="text-5xl md:text-6xl font-black tabular-nums"
                      style={{color: labelColor}}
                    >
                      {result.confidence.toFixed(0)}%
                    </div>
                  </div>
                </div>
                
                <div className="progress-bar h-3">
                  <div 
                    ref={progressRef} 
                    className="progress-fill"
                    style={{
                      width: '0%',
                      background: `linear-gradient(90deg, ${labelColor}, ${labelColor}dd)`
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Metadata Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="glass-card p-5 rounded-xl">
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-blue-500/10">
                    <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div className="text-xs text-gray-400 uppercase tracking-wider">Media Type</div>
                </div>
                <div className="text-lg font-semibold text-gray-200 capitalize">{result.media_type}</div>
              </div>

              <div className="glass-card p-5 rounded-xl">
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-purple-500/10">
                    <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="text-xs text-gray-400 uppercase tracking-wider">Timestamp</div>
                </div>
                <div className="text-sm font-mono text-gray-200">{result.timestamp}</div>
                <div className="text-xs text-gray-500 mt-1">{new Date(result.timestamp).toLocaleString()}</div>
              </div>

              <div className="glass-card p-5 rounded-xl sm:col-span-2">
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-green-500/10">
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                  </div>
                  <div className="text-xs text-gray-400 uppercase tracking-wider">AI Model</div>
                </div>
                <div className="text-sm font-medium text-gray-200">{result.model}</div>
              </div>

              <div className="glass-card p-5 rounded-xl sm:col-span-2">
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-pink-500/10">
                    <svg className="w-5 h-5 text-pink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  </div>
                  <div className="text-xs text-gray-400 uppercase tracking-wider">Location</div>
                </div>
                <div className="text-sm font-medium text-gray-200">{result.geo_tag}</div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-3">
              <button 
                onClick={copyExact}
                className="btn-secondary px-5 py-3 rounded-xl font-medium flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy
              </button>
              <button 
                onClick={downloadReport}
                className="btn-secondary px-5 py-3 rounded-xl font-medium flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download
              </button>
              <button 
                onClick={share}
                className="btn-secondary px-5 py-3 rounded-xl font-medium flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
                Share
              </button>
              <button 
                onClick={onRetry}
                className="ml-auto btn-primary px-6 py-3 rounded-xl font-semibold flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                New Analysis
              </button>
            </div>

            {/* Raw JSON Collapsible */}
            <details className="glass-card rounded-xl overflow-hidden">
              <summary className="cursor-pointer p-5 hover:bg-white/5 transition-colors flex items-center gap-3 font-medium">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
                <span>View Raw JSON Data</span>
              </summary>
              <div className="p-5 bg-black/40 border-t border-white/5">
                <pre className="text-xs text-blue-300 overflow-auto max-h-64 font-mono leading-relaxed">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            </details>
          </div>

          {/* Preview Photo Panel */}
          <div className="w-full lg:w-96 space-y-6">
            <div className="glass-card p-6 rounded-2xl space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm uppercase tracking-wider text-gray-400 font-semibold">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                  Preview Image
                </div>
                <div className={`badge ${badgeClass} text-xs`}>
                  {isDeepfake ? 'Fake' : isReal ? 'Real' : 'Unknown'}
                </div>
              </div>
              
              <div className="gradient-border p-1">
                <div className="bg-black/40 rounded-xl overflow-hidden">
                  <img 
                    src={result.preview_url} 
                    alt="Analysis preview" 
                    className="w-full h-96 object-cover"
                  />
                </div>
              </div>

              {/* Status Summary */}
              <div className="glass-card p-5 rounded-xl text-center space-y-3">
                <div 
                  className="text-4xl font-black uppercase"
                  style={{color: labelColor}}
                >
                  {isDeepfake ? '⚠️ FAKE' : isReal ? '✓ REAL' : '? UNKNOWN'}
                </div>
                <div className="text-sm text-gray-400">
                  Analyzed with {result.confidence.toFixed(1)}% confidence
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Toast Notification */}
      {showToast && (
        <div className="fixed bottom-8 right-8 glass-card px-6 py-4 rounded-xl shadow-2xl animate-slide-in z-50 border-l-4"
             style={{borderColor: '#10b981'}}>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-full bg-green-500/20">
              <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <span className="font-medium text-gray-200">{toastMessage}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export type { DeepfakeResult }
