<?php
    namespace Application\model\classes;

    trait AccessControl
    {
        public function testAccess(array $roles = []) : bool {
            if(!isset($_SESSION['role'])) return false;

            return in_array($_SESSION['role'], $roles) ? true : false;            
        }
    }    
?>