<?php
    namespace Application\Core;

    class App
    {
        public function __construct(
            private string $controller = "", 
            private string $method = "index",
			private string $route = ""
        )
        {
            
        }
        private function splitUrl(): array|string {           
            $url = URL === '/' ? 'home' : URL;
            $url = explode('/', trim($url, "/")); 
            
            if(empty($url[0])) {
                array_shift($url);                
                return $url;
            }                                 

            return $url;
        }

        public function loadController(): void {            
			session_start();
			session_regenerate_id();

			global $id;

			$url = $this->splitUrl();

			// Test diferent options to configure to Controller                         
			if(count($url) == 1 && !empty($url[0])){
				$this->controller = ucfirst($url[0]);
				$this->method = "index";
			}
			else if(count($url) == 2) {
				$this->controller = ucfirst($url[0]);
				$this->method = $url[1];       
			}
			else if(count($url) > 2) {            
				if(!empty($url) && preg_match('/^([0-9]){1,5}$/', $url[count($url) - 1])) {
				$id = $url[count($url) - 1];                                                                     
				array_pop($url);                                                     
				}
				
				foreach ($url as $key => $value) {
				if($key == count($url) - 2) break;
				$this->route .= $value . "/";            
				}                          

				$this->controller = ucfirst($url[count($url) - 2]);
				$this->method = $url[count($url) - 1];                                                       
			} 

			// Build the Controller
			$this->route = "/Application/Controller/" . $this->route;        
			$this->controller = $this->controller . "Controller";

			$file_name = SITE_ROOT . "/.." . $this->route . $this->controller . ".php";

			if(file_exists($file_name)) {                    
				$controller_path = str_replace('/', '\\', $this->route) . $this->controller;                                                         
			} 
			else {                    
				$this->controller = "ErrorController";
				$controller_path = '\Application\Controller\\' . ucfirst($this->controller);								
			} 

			$controller = new $controller_path;

			/** select method */
			if(count($url) > 0) {				
                if(method_exists($controller, $this->method)) {                        
                	array_shift($url);
                }
				else {
					$this->method = "index";
				}
			}

			call_user_func_array([$controller, $this->method], []);
        }
    }    
?>
