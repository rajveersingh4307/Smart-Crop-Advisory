import React from "react";

const DiagnosisResultCard = ({ diagnosis, isDarkMode }) => {
    if (!diagnosis) return null;

    const { confidenceScore, treatment, classId } = diagnosis;

    const ulClass = `list-none space-y-3 p-4 rounded-lg ${
        isDarkMode ? "bg-gray-700" : "bg-gray-100"
    }`;

    return (
        <div className="space-y-6">
            <div className="p-5 rounded-xl shadow-lg border-l-8 border-emerald-500 bg-emerald-500/10">
                <h3 className="text-3xl font-extrabold flex items-center mb-2">
                    🌿 {diagnosis.diagnosis}
                </h3>

                <div className="flex justify-between items-center text-lg mt-3 pt-3 border-t border-gray-600/50">
                    <div>
                        <span className="font-semibold block text-sm text-gray-400">
                            Model Confidence
                        </span>
                        <span className="text-2xl font-bold text-emerald-400">
                            {confidenceScore}
                        </span>
                    </div>

                    <div>
                        <span className="font-semibold block text-sm text-gray-400">
                            Class ID
                        </span>
                        <span className="text-2xl font-bold text-blue-400">
                            {classId}
                        </span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                    <h4 className="text-xl font-bold text-red-400 mb-3 flex items-center">
                        <span className="text-2xl mr-2">💊</span>
                        Immediate Action
                    </h4>

                    <ul className={ulClass}>
                        {treatment?.immediate?.map((item, index) => (
                            <li key={index} className="flex items-start text-sm">
                                <span className="text-red-500 mr-2 flex-shrink-0">•</span>
                                {item}
                            </li>
                        ))}
                    </ul>
                </div>

                <div>
                    <h4 className="text-xl font-bold text-blue-400 mb-3 flex items-center">
                        <span className="text-2xl mr-2">🌱</span>
                        Preventive Measures
                    </h4>

                    <ul className={ulClass}>
                        {treatment?.preventive?.map((item, index) => (
                            <li key={index} className="flex items-start text-sm">
                                <span className="text-emerald-500 mr-2 flex-shrink-0">✓</span>
                                {item}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>

            <p className="text-xs text-gray-500">
                Detection is based on the trained EfficientNet-B0 model. Use the
                result as decision support and confirm uncertain cases with an
                agricultural expert.
            </p>
        </div>
    );
};

export default DiagnosisResultCard;
