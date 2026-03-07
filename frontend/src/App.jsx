import {useCallback, useEffect, useState} from 'react'
import Canvas from "./Canvas.jsx"
import ProbabilityBar from "./Probs.jsx"

function App() {
    const [probs, setProbs] = useState(Array(10).fill(0))
    const [models, setModels] = useState([])
    const [selectedModel, setSelectedModel] = useState("")
    const [modelInfo, setModelInfo] = useState(null);

    useEffect(() => {
        let interval = setInterval(() => {
            if (window.pywebview) {
                window.pywebview.api.get_models().then(result => {
                    setModels(result);
                    window.pywebview.api.get_current_model().then(async current => {
                        setSelectedModel(current ?? (result[0] ?? ""));
                        if (current ?? result[0]) {
                            const info = await window.pywebview.api.get_model_info();
                            setModelInfo(info)
                        }
                    });
                });
                clearInterval(interval);
            }
        }, 100);
        return () => clearInterval(interval);
    }, []);



    // Modellwechsel an Backend melden
    const handleModelChange = async (e) => {
        const model = e.target.value;
        setSelectedModel(model);
        if (window.pywebview) {
            await window.pywebview.api.set_model(model);
            const info = await window.pywebview.api.get_model_info();
            setModelInfo(info);
        }
        setProbs(Array(10).fill(0)); // Reset Probbars
    };

    const setProbabilities = (result) => {
        setProbs(result)
    }

    const handleVector = useCallback(async (vector) => {
        if (window.pywebview) {
            const result = await window.pywebview.api.predict_digit(vector);
            const arr = Array.isArray(result[0]) ? result[0] : result;
            if (Array.isArray(arr) && arr.length === 10) {
                setProbabilities(arr.map(x => x*100));
            }
            console.log(arr);
        }
    }, []);

    return (
        <div className="min-h-screen flex items-center justify-center bg-[#242424] text-[rgba(255,255,255,0.87)]">
            <div className="mr-8 p-6 border border-white/20 rounded-2xl bg-black flex-shrink-0">
                <Canvas onVector={handleVector} />
                <div className="flex flex-col items-center mt-4 w-full">
                    <select
                        value={selectedModel}
                        onChange={handleModelChange}
                        className="mt-2 px-3 py-2 rounded border border-gray-600 bg-[#181818] text-white w-full"
                    >
                        {models.map((model) => (
                            <option key={model} value={model}>{model}</option>
                        ))}
                    </select>
                    
                    {modelInfo && (
                        <div className="mt-4 p-3 rounded bg-[#181818] border border-gray-700 text-sm w-full">
                            <div>Parameters: {modelInfo.parameters}</div>
                            <div>Layers: [{modelInfo.layers?.join(", ")}]</div>
                            <div>
                                Training params: Epochs: {modelInfo.epochs ?? "?"},
                                learnrate: {modelInfo.learning_rate ?? "?"},
                                batchsize: {modelInfo.batch_size ?? "?"}
                            </div>
                            <div>
                                Acc: Learn Acc: {modelInfo.train_acc?.toFixed(2) ?? "?"}%,
                                Test Acc: {modelInfo.test_acc?.toFixed(2) ?? "?"}%
                            </div>
                        </div>
                    )}
                    
                </div>
            </div>
            <div className="p-6 border border-white/20 rounded-2xl bg-black flex flex-col gap-0 w-72">
                {probs.map((percentage, digit) => (
                    <ProbabilityBar
                        key={digit}
                        value={digit}
                        percentage={percentage}
                        isLast={digit === probs.length - 1}
                    />
                ))}
            </div>
        </div>
    )
}

export default App