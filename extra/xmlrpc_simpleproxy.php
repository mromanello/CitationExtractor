<?php
# using the library at http://keithdevens.com/software/xmlrpc
$XMLRPC_LIB="/home/56k/xmlrpc.php";
$XMLRPC_URL="www.mr56k.info:8001";
$TEST_STRING="Hom. Il. 1.1 is a canonical citation";

# include the dependency
include($XMLRPC_LIB);
define("XMLRPC_DEBUG", 1);
$test=XMLRPC_request($XMLRPC_URL, "/rpc/crex", "json",array(XMLRPC_prepare($TEST_STRING)));
print($test[1]);
#XMLRPC_debug_print();
?>