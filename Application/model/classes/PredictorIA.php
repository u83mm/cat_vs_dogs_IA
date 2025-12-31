<?php
declare(strict_types = 1);

namespace Application\model\classes;

class PredictorIA
{
    private const AI_SERVICE_URL = "http://ai-service:5000/predict";

    public static function getPrediction(string $file_name) : array
    {
        $data = json_encode(['filename' => $file_name]);

        $ch = curl_init(self::AI_SERVICE_URL);

        // Configuramos opciones de forma moderna
        curl_setopt_array($ch, [
            CURLOPT_POST           => true,
            CURLOPT_POSTFIELDS     => $data,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER     => [
                'Content-Type: application/json',
                'Content-Length: ' . strlen($data)
            ],
            // Timeout de 10 segundos por si la GPU está ocupada
            CURLOPT_CONNECTTIMEOUT => 5,
            CURLOPT_TIMEOUT        => 10,
        ]);

        $response = curl_exec($ch);
        $statusCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        if($error) {
            return [
                'status' => 'error',
                'message' => 'conection error with AI: ' . $error, 
            ];
        }

        if ($statusCode !== 200) {
            return [
                'status' => 'error', 
                'error'  => "AI service returned $statusCode",
                'details' => $response
            ];
        }

        return json_decode($response, true) ?? [
            'status' => 'error',
            'message' => 'invalid JSON response from AI service',
        ];
    }
}