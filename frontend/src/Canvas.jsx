import { useRef, useEffect } from "react";

const CANVAS_SIZE = 280;
const GRID_SIZE = 28;
const SCALE = CANVAS_SIZE / GRID_SIZE;

export default function Canvas({ onVector }) {
    const canvasRef = useRef(null);
    const isDrawing = useRef(false);
    const isErasing = useRef(false);
    const ctxRef = useRef(null);
    const runRealtimeInferenceRef = useRef(null);
    
    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = 16;
        ctx.strokeStyle = "white";
        
        ctxRef.current = ctx
    }, []);

    useEffect(() => {
        runRealtimeInferenceRef.current = throttle(() => {
            const vector = getVector();
            if (onVector) onVector(vector);
        }, 60);
    }, [onVector]);
    
    const getMousePos = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        return {
            x: ((e.clientX - rect.left) / rect.width) * CANVAS_SIZE,
            y: ((e.clientY - rect.top) / rect.height) * CANVAS_SIZE,
        };
    };
    
    const handleMouseDown = (e) => {
        e.preventDefault();
        isDrawing.current = true;
        isErasing.current = e.button === 2;
        
        const ctx = ctxRef.current;
        ctx.beginPath();
        
        const { x, y } = getMousePos(e);
        ctx.moveTo(x, y);
    }
    
    const handleMouseMove = (e) => {
        if (!isDrawing.current) return;
        
        const ctx = ctxRef.current;
        const { x, y } = getMousePos(e);
        
        ctx.strokeStyle = isErasing.current ? "black" : "white";
        ctx.lineTo(x, y);
        ctx.stroke();
        
        runRealtimeInferenceRef.current?.();
    }
    
    const handleMouseUp = () => {
        isDrawing.current = false;
    }
    
    const clearCanvas = () => {
        const ctx = ctxRef.current;
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }
    
    const getVector = () => {
        const ctx = ctxRef.current;
        const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data;
        
        const vector = [];
        
        for (let gy = 0; gy < GRID_SIZE; gy++) {
            for (let gx = 0; gx < GRID_SIZE; gx++) {
                let sum = 0;
                
                for (let y = 0; y < SCALE; y++) {
                    for (let x = 0; x < SCALE; x++) {
                        const px = (gy * SCALE + y) * CANVAS_SIZE + (gx * SCALE + x);
                        const idx = px * 4;
                        sum += imageData[idx];
                    }
                }
                
                const avg = sum / (SCALE * SCALE);
                vector.push(avg / 255);
            }
        }
        return vector;
    };
    
    function throttle(fn, limit) {
        let inThrottle = false;
        return function (...args) {
            if (!inThrottle) {
                fn.apply(this, args);
                inThrottle = true;
                setTimeout(() => (inThrottle = false), limit);
            }
        }
    }
    
    
    
    return (
        <div className="flex flex-col items-center select-none" onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} onContextMenu={(e) => e.preventDefault()}>
            <canvas ref={canvasRef} width={CANVAS_SIZE} height={CANVAS_SIZE} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} className="border border-gray-600 cursor-crosshair"/>
            <div className="flex gap-4 mt-4">
                <button onClick={clearCanvas} className="px-4 py-2 bg-red-600 text-white rounded">
                    Clear
                </button>
            </div>
        </div>
    );
}