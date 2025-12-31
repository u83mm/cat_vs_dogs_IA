<?php

    declare(strict_types = 1);

    namespace Application\Controller;

    use Application\Core\Controller;

    class HomeController extends Controller
    {        
        public function __construct(private object $dbcon = DB_CON)
        {

        }

        public function index()
        { 
            try {                                                          
                $this->render('main_view.twig', [
                    'menus' => $this->showNavLinks(), 
                    'session'   =>  $_SESSION,
                    'active'    =>  'home',                
                ]);

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
?>  
