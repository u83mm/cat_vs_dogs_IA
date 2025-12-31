<?php

    declare(strict_types = 1);

    namespace Application\Controller;

    use Application\Core\Controller;
    use Application\model\classes\Query;
    use Application\model\classes\Validate;

    class RegisterController extends Controller 
    {
        public function __construct(private object $dbcon = DB_CON) {
                        
        }

        /** Show register view */
        public function index(): void {
            try {
                $this->render('register/register_view.twig', [
                    'menus'     =>  $this->showNavLinks(),
                    'session'   =>  $_SESSION,
                    'active'    =>  'registration',
                ]);

            } catch (\Throwable $th) {
                $error_msg = [
                    'error' =>  $th->getMessage(),
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

        /** Register a new user */
        public function new(): void {
            $validate = new Validate;
            $query = new Query;             
                        
            try {                                
                if($_SERVER['REQUEST_METHOD'] == 'POST') { 
                    
                    // Get values from register form                  
                    $fields = [
                        'user_name' =>  $validate->test_input(strtolower($_REQUEST['user_name'])),
                        'email'     =>  $validate->validate_email($_REQUEST['email']) ? $validate->test_input(strtolower($_REQUEST['email'])): null,
                        'password'  =>  $_REQUEST['password'] === $_REQUEST['repeat_password'] ? $validate->test_input($_REQUEST['password']) : "",
                    ];                    
                    
                    // Test if the e-mail is in use by other user
                    $result = $query->selectOneBy('users', 'email', $fields['email'], $this->dbcon);                    

                    if($result) {
                        $this->render('register/register_view.twig', [
                            'error_message' =>  'The user is already in use.',
                            'fields'        =>  $fields,
                            'menus'         =>  $this->showNavLinks(),
                            'active'        =>  'registration',
                        ]);
                    }
                    else {
                        // Test if passwords are equals
                        if(empty($fields['password'])) {
                            $fields['password'] = $validate->test_input($_REQUEST['password']);

                            $this->render('register/register_view.twig', [
                                'menus'         =>  $this->showNavLinks(),
                                'error_message' =>  "Passwords are not equals", 
                                'fields'        =>  $fields,
                                'active'        =>  'registration',              
                            ]);                         
                        }
                        else {
                            // Validate form
                            $ok = $validate->validate_form($fields);
                            
                            if($ok) {                        
                                // Register the user
                                $query->insertInto('users', $fields, $this->dbcon);
    
                                $this->render('register/register_view.twig', [
                                    'menus'         =>  $this->showNavLinks(),
                                    'message'       =>  "User registered successfully",
                                    'active'        =>  'registration',                                                  
                                ]);                                                
                            } 
                        }
                    }                                                                                                                                                                          
                }
                else {
                    throw new \Exception("Service unavailable", 1);                    
                }                                
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