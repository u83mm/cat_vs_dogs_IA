<?php

    declare(strict_types = 1);

    namespace Application\Controller;

	use Application\Core\Controller;

    class LogoutController extends Controller
    {
        /* Unsetting the session variables and destroying the session. */
        public function index(): void
        {
            unset($_SESSION['id_user']);
			unset($_SESSION['user_name']);
			unset($_SESSION['role']);			
		  
			$_SESSION = array();
		  
			session_destroy();
			setcookie('PHPSESSID', "0", time() - 3600);		  			            

			$this->render('main_view.twig', [
                'menus'     =>  $this->showNavLinks(),
                'session'   =>  $_SESSION, 
                'active'    =>  'home',                                                                    
            ]);
        }
    }    
?>
