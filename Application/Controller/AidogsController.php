<?php
declare(strict_types = 1);

namespace Application\Controller;

use Application\Core\Controller;
use Application\model\classes\PredictorIA;

final class AidogsController extends Controller
{
    public function __construct(private object $dbcon = DB_CON)
    {

    }

    public function index()
    { 
        try {            
            $predictorObj = new PredictorIA();
                        
            $twig_variables = [
                'menus'     =>  $this->showNavLinks(), 
                'session'   =>  $_SESSION,
                'active'    =>  'Ai-dogs',                
            ];

            if($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['mascota'])) {
                $image = $_FILES['mascota']['name'];

                // Rename the file to avoid conflicts
                //$image = time() . '_' . preg_replace('/\s+/', '_', $image);
                $image = uniqid('img_', true) . '.' . pathinfo($image, PATHINFO_EXTENSION);

                // Move the uploaded file to the target directory
                $target_dir = SITE_ROOT .'/../ai_temp_uploads/';
                $target_file = $target_dir . basename($image);

                if(move_uploaded_file($_FILES['mascota']['tmp_name'], $target_file)) {
                    $result = $predictorObj->getPrediction($image);
                    unlink($target_file);

                    if(($result['status'] ?? '') === 'success') {
                        $twig_variables = array_merge($twig_variables, [
                            'result'   =>  $result['class'] ?? [],
                            'confidence'   =>  $result['confidence'] ?? 0,                                            
                        ]);                
                    }
                    else {
                        $twig_variables = array_merge($twig_variables, [
                            'error_message' =>  $result['message'] ?? 'An error occurred',
                        ]);                
                    }
                }
                else {
                    $twig_variables = array_merge($twig_variables, [
                        'error_message' =>  'Failed to upload image.',
                    ]);                
                }                                                                                                
            }                        

            $this->render('ai/ai_dogs_view.twig', $twig_variables);            

        } catch (\Throwable $th) {
            $error_msg = [
                'Error' =>  $th->getMessage(),
            ];

            if(isset($_SESSION['role']) && $_SESSION['role'] === 'ROLE_ADMIN') {
                $error_msg = [
                    "Message:"  =>  $th->getMessage(),
                    "Path:"     =>  $th->getFile(),
                    "Line:"     =>  $th->getLine(),
                ];
            }

            $this->render('error_view.twig', [
                'menus'             => $this->showNavLinks(),
                'exception_message' => $error_msg,                
            ]);
        }                                             
    }
}
