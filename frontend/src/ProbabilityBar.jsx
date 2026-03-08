export default function ProbabilityBar({ value, percentage, isLast }) {
    const limited = Math.max(0, Math.min(percentage ?? 0, 100))
    const rounded = limited.toFixed(2)

    return (
        <div className="w-full">
            <div className="flex items-center h-8">
                <span className="text-white font-semibold text-lg w-6 text-right mr-3 select-none">{value}</span>
                <div className="relative flex-1 h-4 bg-white/10 rounded-full overflow-hidden mx-3">
                    <div
                        className="absolute left-0 top-0 h-full bg-linear-to-r from-blue-400 to-blue-600 rounded-full transition-all duration-500 ease-out"
                        style={{ width: `${limited}%` }}
                    />
                </div>
                <span className="text-white font-mono text-base w-16 text-left ml-3 select-none">{rounded}%</span>
            </div>
            {!isLast && (
                <div className="border-b border-white/10 my-2" />
            )}
        </div>
    );
}