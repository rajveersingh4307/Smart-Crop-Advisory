import React, { useState, useRef } from 'react';
import PageWrapper from '../../layout/PageWrapper';
import DiagnosisResultCard from './DiagnosisResultCard';

const ML_API_URL = "http://127.0.0.1:8000/predict";

const DiagnosisPage = ({ isDarkMode }) => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [diagnosisResult, setDiagnosisResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const resetDiagnosis = (fullReset = true) => {
        setSelectedImage(null);
        setDiagnosisResult(null);
        setError(null);
        if (fullReset && fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const handleImageUpload = (event) => {
        const file = event.target.files?.[0];

        if (!file) return;

        if (!file.type.startsWith("image/")) {
            setError("Please upload a valid image file (JPEG or PNG).");
            setSelectedImage(null);
            setDiagnosisResult(null);
            return;
        }

        setError(null);
        setDiagnosisResult(null);
        setSelectedImage(file);
    };

    const callDiagnosisApi = async () => {
        if (!selectedImage) {
            setError("Please select an image first.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setDiagnosisResult(null);

        try {
            const formData = new FormData();
            formData.append("file", selectedImage);

            const response = await fetch(ML_API_URL, {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.message || `Prediction failed with status ${response.status}.`);
            }

            setDiagnosisResult(data.result);
        } catch (err) {
            console.error("Disease detection error:", err);
            setError(
                "Could not connect to the disease detection model. Make sure the AgroCare ML backend is running on port 8000."
            );
        } finally {
            setIsLoading(false);
        }
    };

    const handleTriggerUpload = () => {
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
            fileInputRef.current.click();
        }
    };

    const imagePreviewUrl = selectedImage
        ? URL.createObjectURL(selectedImage)
        : null;

    const buttonClass =
        "w-full py-3 font-semibold rounded-lg transition duration-300 shadow-md";

    return (
        <PageWrapper title="AI Leaf Diagnosis" isDarkMode={isDarkMode}>
            <div className="mb-8">
                <input
                    type="file"
                    accept="image/jpeg,image/png"
                    onChange={handleImageUpload}
                    ref={fileInputRef}
                    className="hidden"
                    disabled={isLoading}
                />

                {imagePreviewUrl ? (
                    <div className="relative border-4 border-emerald-500 rounded-xl overflow-hidden shadow-lg">
                        <img
                            src={imagePreviewUrl}
                            alt="Selected Leaf"
                            className="w-full h-80 object-contain p-2 bg-gray-100 dark:bg-gray-700"
                        />

                        <div className="absolute top-2 right-2 flex space-x-2">
                            <button
                                onClick={() => resetDiagnosis(true)}
                                className="p-2 bg-red-600 text-white rounded-full hover:bg-red-700 transition"
                                disabled={isLoading}
                            >
                                ❌
                            </button>

                            <button
                                onClick={handleTriggerUpload}
                                className="p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition"
                                disabled={isLoading}
                            >
                                🔄
                            </button>
                        </div>
                    </div>
                ) : (
                    <div
                        onClick={handleTriggerUpload}
                        className="p-6 rounded-xl border-2 border-dashed border-emerald-500 bg-emerald-500/10 h-64 flex flex-col items-center justify-center cursor-pointer hover:bg-emerald-500/20 transition"
                    >
                        <span className="text-6xl mb-2">📸</span>
                        <span className="text-xl font-semibold text-emerald-400">
                            Click to Upload Image
                        </span>
                        <span className="text-sm mt-1 text-gray-500">
                            Supported: JPEG, PNG
                        </span>
                    </div>
                )}
            </div>

            <div className="flex space-x-4 mb-8">
                <button
                    onClick={callDiagnosisApi}
                    disabled={!selectedImage || isLoading}
                    className={`${buttonClass} ${
                        selectedImage && !isLoading
                            ? "bg-emerald-500 hover:bg-emerald-600 text-white"
                            : "bg-gray-400 text-gray-700 cursor-not-allowed"
                    }`}
                >
                    {isLoading ? "Analyzing Leaf..." : "Get Diagnosis & Treatment"}
                </button>
            </div>

            <div className="mt-8 pt-6 border-t border-gray-700/50">
                {error && (
                    <div className="p-4 bg-red-600/10 text-red-400 rounded-lg border border-red-500">
                        {error}
                    </div>
                )}

                {diagnosisResult && (
                    <DiagnosisResultCard
                        diagnosis={diagnosisResult}
                        isDarkMode={isDarkMode}
                    />
                )}

                {!selectedImage && !diagnosisResult && !error && (
                    <p className="text-gray-500 italic">
                        Upload a potato or tomato leaf image to get disease detection and treatment guidance.
                    </p>
                )}
            </div>
        </PageWrapper>
    );
};

export default DiagnosisPage;
