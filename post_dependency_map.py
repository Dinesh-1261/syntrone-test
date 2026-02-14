# dependencies/post_dependency_map.py

POST_DEPENDENCY_MAP = {

  
   

    # --------------------------------------------------
   
    # --------------------------------------------------
    # VALUE HELP / OPTIONS
    # --------------------------------------------------
    "/options/add": {
        "method": "POST",
        "fields": {
            "type": {
                "label": "Type",
                "source": "static",
                "key": "option_type"
            },
            "label": {
                "label": "Label"
            },
            "value": {
                "label": "Value"
            }
        }
    },
   # --------------------------------------------------
# SALES QUOTATION → DOWNLOAD PDF
# --------------------------------------------------
"/sales-quotations/download": {
    "method": "POST",
    "fields": {
        "quotationNumber": {
            "label": "Quotation Number"
        },
        "printType": {
            "label": "Print Type",
            "source": "static",
            "key": "sales_quotation_print_type"
        }
    }
},
# --------------------------------------------------
# SALES DELIVERY → DOWNLOAD PDF
# --------------------------------------------------
"/sales-delivery/download": {
    "method": "POST",
    "fields": {
        "deliveryNo": {
            "label": "Sales Delivery Number"
        },
        "printType": {
            "label": "Print Type",
            "source": "static",
            "key": "sales_delivery_print_type"
        }
    }
},
# --------------------------------------------------
# DELIVERY RETURN → DOWNLOAD PDF
# --------------------------------------------------
"/test/delivery-returns/download": {
    "method": "POST",
    "fields": {
        "returnNo": {
            "label": "Delivery Return Number"
        },
        "printType": {
            "label": "Print Type",
            "source": "static",
            "key": "delivery_return_print_type"
        }
    }
},
# --------------------------------------------------
# CUSTOMER INVOICE RECEIPT → DOWNLOAD PDF
# --------------------------------------------------
"/customer-invoice-receipts/download": {
    "method": "POST",
    "fields": {
        "receiptNo": {
            "label": "Customer Invoice Receipt Number",
            "required": True
        },
        "printType": {
            "label": "Print Type",
            "source": "static",
            "key": "invoice_receipt_print_type",
            "required": True
        }
    }
},
# --------------------------------------------------
# DIRECT RECEIPT → DOWNLOAD PDF
# --------------------------------------------------
"/direct-receipts/download": {
    "method": "POST",
    "fields": {
        "receiptNo": {
            "label": "Direct Receipt Number"
        },
        "printType": {
            "label": "Print Type",
            "source": "static",
            "key": "direct_receipt_print_type"
        }
    }
},


}
