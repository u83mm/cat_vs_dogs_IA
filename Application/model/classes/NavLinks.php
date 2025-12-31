<?php
    namespace Application\model\classes;

    trait NavLinks
    {
        public function __construct(private array $menus = [])
        {
            
        }

        public function showAdminLinks(): array
        {
            $this->menus = [
                "Home"			  => "/",				
				"Registration"	  => "/register",
				"Administration"  => "/admin/admin",				
				"Login"			  => "/login",
            ];

            return $this->menus;
        }

        
        public function showUserLinks(): array
        {
            $this->menus = [
                "Home"			=> "/",
                "Ai-dogs"       => "/aidogs",				
				"Registration"	=> "/register",				
				"Login"			=> "/login",
            ];

            return $this->menus;
        }
    }    
?>
