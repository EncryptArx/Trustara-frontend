import React, {useRef, useState} from 'react'
import { detectDeepfake } from './mockApi'
import { DeepfakeResult } from './types'

export default function Uploader({ onResult }: { onResult: (r: DeepfakeResult)=>void }){
  const inputRef = useRef<HTMLInputElement|null>(null)
  const [file, setFile] = useState<File| null>(null)
  const [preview, setPreview] = useState<string| null>(null)
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [metadata, setMetadata] = useState<{name: string, size: string, type: string} | null>(null)

  function onFile(f?: File){
    if(!f) return
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreview(url)
    setMetadata({
      name: f.name,
      size: `${(f.size / 1024 / 1024).toFixed(2)} MB`,
      type: f.type
    })
  }

  async function analyze(){
    if (!file) return
    setLoading(true)
    try {
      const res = await detectDeepfake(file)
      onResult(res)
    } finally {
      setLoading(false)
    }
  }

  function handleDrop(e: React.DragEvent){
    e.preventDefault()
    e.stopPropagation()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if(f && f.type.startsWith('image/')) onFile(f)
  }

  function handleDragOver(e: React.DragEvent){
    e.preventDefault()
    e.stopPropagation()
  }

  return (
    <div className="glass-card p-8 md:p-12 rounded-3xl animate-scale-in">
      <div className="flex flex-col lg:flex-row gap-10 items-stretch">
        {/* Upload Zone */}
        <div className="flex-1 w-full">
          <div
            onClick={()=>inputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragEnter={()=>setDragging(true)}
            onDragLeave={()=>setDragging(false)}
            className={`
              relative border-2 border-dashed rounded-2xl p-12 cursor-pointer 
              transition-all duration-300 overflow-hidden
              ${dragging ? 'drag-active' : 'border-gray-700 hover:border-cyan-500/50'}
              ${preview ? 'bg-black/20' : 'bg-gray-900/30'}
            `}
            role="button"
            tabIndex={0}
            aria-label="Upload image"
            onKeyDown={(e)=>{ if(e.key === 'Enter' || e.key === ' ') inputRef.current?.click() }}
          >
            <input 
              ref={inputRef} 
              type="file" 
              accept="image/*" 
              hidden 
              onChange={(e)=>onFile(e.target.files?.[0])} 
            />
            
            {!preview ? (
              <div className="text-center py-8">
                <div className="mb-6 inline-flex p-6 rounded-full bg-gradient-to-br from-blue-500/10 to-purple-500/10">
                  <svg className="w-20 h-20 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <div className="text-xl font-semibold text-gray-200 mb-2">
                  Drag & Drop Your Image
                </div>
                <div className="text-gray-400 mb-6">
                  or click to browse from your device
                </div>
                <div className="flex items-center justify-center gap-4 text-xs text-gray-500">
                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                    </svg>
                    PNG, JPG, JPEG
                  </span>
                  <span>•</span>
                  <span>Max 10MB</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="gradient-border p-1">
                  <div className="bg-black/40 rounded-xl p-2">
                    <img 
                      src={preview} 
                      alt="preview" 
                      className="max-h-72 w-full mx-auto rounded-lg object-contain"
                    />
                  </div>
                </div>
                {metadata && (
                  <div className="text-center space-y-2 animate-slide-in">
                    <div className="text-gray-200 font-medium truncate px-4">{metadata.name}</div>
                    <div className="flex items-center justify-center gap-3 text-sm text-gray-400">
                      <span>{metadata.size}</span>
                      <span>•</span>
                      <span>{metadata.type.split('/')[1].toUpperCase()}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Action Panel */}
        <div className="w-full lg:w-96 flex flex-col gap-6">
          <div className="space-y-4">
            <button 
              className="btn-secondary w-full px-8 py-4 rounded-xl font-semibold text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={()=>inputRef.current?.click()}
              disabled={loading}
            >
              <span className="flex items-center justify-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                Select File
              </span>
            </button>
            
            <button 
              className="btn-primary w-full px-8 py-5 rounded-xl font-bold text-white text-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
              onClick={analyze} 
              aria-busy={loading} 
              disabled={loading || !file}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-3">
                  <div className="spinner" />
                  <span>Analyzing...</span>
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Start Analysis
                </span>
              )}
            </button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-3">
            <div className="glass-card p-4 rounded-xl text-center">
              <div className="text-2xl font-bold text-blue-400">99.8%</div>
              <div className="text-xs text-gray-500 mt-1">Accuracy</div>
            </div>
            <div className="glass-card p-4 rounded-xl text-center">
              <div className="text-2xl font-bold text-purple-400">&lt;2s</div>
              <div className="text-xs text-gray-500 mt-1">Processing</div>
            </div>
          </div>

          {/* Info Cards */}
          <div className="space-y-3">
            <div className="glass-card p-4 rounded-xl">
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-blue-500/10">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <div className="text-sm font-semibold text-gray-200">AI-Powered Detection</div>
                  <div className="text-xs text-gray-400 mt-1">Advanced neural networks analyze facial patterns</div>
                </div>
              </div>
            </div>

            <div className="glass-card p-4 rounded-xl">
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-green-500/10">
                  <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <div className="text-sm font-semibold text-gray-200">100% Private</div>
                  <div className="text-xs text-gray-400 mt-1">No data stored, processed locally in browser</div>
                </div>
              </div>
            </div>

            <div className="glass-card p-4 rounded-xl">
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-purple-500/10">
                  <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <div className="text-sm font-semibold text-gray-200">Lightning Fast</div>
                  <div className="text-xs text-gray-400 mt-1">Get results in under 2 seconds</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
