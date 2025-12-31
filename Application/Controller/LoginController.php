<?php

    declare(strict_types = 1);

    namespace Application\Controller;

    use Application\Core\Controller;
    use Application\model\classes\Query;
    use Application\model\classes\Validate;

    class LoginController extends Controller
    {        
        public function __construct(
            private object $dbcon = DB_CON,
            private Query $query = new Query    
        )
        {

        }

        public function index(): void
        { 
            $validate = new Validate;            

            try {
                if($_SERVER['REQUEST_METHOD'] === 'POST') {
                    // Get values from login form
                    $fields = [
                        'email'     =>  $validate->validate_email(strtolower($_REQUEST['email'])) ? $validate->test_input(strtolower($_REQUEST['email'])) : "",
                        'password'  =>  $validate->test_input($_REQUEST['password']) ?? "",
                    ];

                    // Validate form                    
                    if($validate->validate_form($fields)) {
                        if(!isset($_SESSION['id_user'])) {
                            // Test user to do login                                                                                                                                                                  
                            if($result = $this->query->selectOneByFieldNameInnerjoinOnfield(
                                            'users', 
                                            'roles', 
                                            'id_role', 
                                            'email', 
                                            $fields['email']
                                        )                            
                            ) {                                
                                if(password_verify($fields['password'], $result['password'])) {												
                                    $_SESSION['id_user']    = $result['id'];						
                                    $_SESSION['user_name']  = $result['user_name'];
                                    $_SESSION['role']       = $result['role'];												
                                                                                                       
                                    $this->render('main_view.twig', [
                                        'menus'     =>  $this->showNavLinks(),                                     
                                        'session'   =>  $_SESSION, 
                                        'active'    =>  'home',                  
                                    ]);						
                                }
                                else {
                                    $this->render('login/login_view.twig', [
                                        'menus'         =>  $this->showNavLinks(), 
                                        'error_message' =>  'Please test your credentials', 
                                        'fields'        =>  $fields,
                                        'active'        =>  'login',                   
                                    ]);
                                }                                                               
                            }
                            else {
                                $this->render('login/login_view.twig', [
                                    'menus'         =>  $this->showNavLinks(), 
                                    'error_message' =>  'Please test your credentials',
                                    'fields'        =>  $fields, 
                                    'active'        =>  'login',                
                                ]);
                            }                           
                        }
                        else {
                            $this->render('main_view.twig', [
                                'menus'     =>  $this->showNavLinks(),                                     
                                'session'   =>  $_SESSION,                  
                            ]);	
                        }
                    }
                }                                                                               

                $this->render('login/login_view.twig', [
                    'menus'     => $this->showNavLinks(),
                    'active'    =>  'login',                
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
