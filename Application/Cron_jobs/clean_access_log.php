<?php 
    // Define the path to file
    $path = SITE_ROOT . "/../Application/log/apache/access.log";

    // Read the content's file
    if(file_exists($path)) {
        $content_array = file($path);

        // Build an array with the last 20 lines
        $new_content_array = array_slice($content_array, count($content_array) - 20);                 

        // Create a new file with the last 20 lines in the array.
        $file = fopen($path,"w");

        foreach ($new_content_array as $key => $line) {
            fwrite($file, $line);
        }

        fclose($file);                                           
    }
?>