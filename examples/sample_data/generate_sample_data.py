"""Generate realistic financial document samples."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from semantic_search.models.document import Document, DocumentType


def generate_invoices(count: int = 20) -> List[Document]:
    """Generate sample invoice documents."""
    companies = [
        "ACME Corporation", "TechCorp Solutions", "Global Services Inc",
        "Digital Dynamics", "Innovation Labs", "Enterprise Systems",
        "Cloud Nine Technologies", "Quantum Solutions", "Alpha Industries"
    ]
    
    services = [
        "Web development and design services",
        "Software consulting and implementation",
        "Cloud infrastructure management",
        "Database optimization and maintenance",
        "Mobile application development",
        "System integration services",
        "IT support and maintenance",
        "Digital marketing services",
        "Business process automation"
    ]
    
    invoices = []
    base_date = datetime.now() - timedelta(days=180)
    
    for i in range(count):
        company = random.choice(companies)
        service = random.choice(services)
        amount = round(random.uniform(500, 15000), 2)
        invoice_date = base_date + timedelta(days=random.randint(0, 180))
        due_date = invoice_date + timedelta(days=30)
        
        content = f"""
        INVOICE #{f'INV-2024-{i+1:03d}'}
        
        From: {company}
        Date: {invoice_date.strftime('%Y-%m-%d')}
        Due Date: {due_date.strftime('%Y-%m-%d')}
        
        Description: {service}
        Amount: ${amount:,.2f}
        Tax (8.5%): ${amount * 0.085:,.2f}
        Total Amount: ${amount * 1.085:,.2f}
        
        Payment Terms: Net 30 days
        Payment Method: Bank transfer or check
        
        Thank you for your business!
        """.strip()
        
        doc = Document(
            id=f"inv_{i+1:03d}",
            content=content,
            entity_id=company.lower().replace(" ", "_").replace(".", ""),
            doc_type=DocumentType.INVOICE,
            date=invoice_date,
            metadata={
                "amount": amount * 1.085,
                "tax_rate": 0.085,
                "due_date": due_date.isoformat(),
                "service_category": service.split()[0].lower()
            }
        )
        invoices.append(doc)
    
    return invoices


def generate_purchase_orders(count: int = 15) -> List[Document]:
    """Generate sample purchase order documents."""
    vendors = [
        "Office Depot", "Staples Inc", "Amazon Business",
        "Dell Technologies", "HP Enterprise", "Microsoft Corporation",
        "Adobe Systems", "Salesforce", "Oracle Corporation"
    ]
    
    items = [
        ("Office supplies (paper, pens, folders)", 150, 350),
        ("Computer hardware (laptops, monitors)", 800, 2500),
        ("Software licenses (productivity suite)", 200, 1200),
        ("Furniture (desks, chairs, cabinets)", 300, 1500),
        ("Networking equipment (routers, switches)", 400, 2000),
        ("Printing supplies (toner, paper)", 100, 400),
        ("Mobile devices (phones, tablets)", 500, 1800),
        ("Cloud services subscription", 100, 800),
        ("Training and certification materials", 200, 1000)
    ]
    
    pos = []
    base_date = datetime.now() - timedelta(days=120)
    
    for i in range(count):
        vendor = random.choice(vendors)
        item_desc, min_amount, max_amount = random.choice(items)
        amount = round(random.uniform(min_amount, max_amount), 2)
        po_date = base_date + timedelta(days=random.randint(0, 120))
        delivery_date = po_date + timedelta(days=random.randint(5, 21))
        
        content = f"""
        PURCHASE ORDER #{f'PO-2024-{i+1:03d}'}
        
        Vendor: {vendor}
        Date: {po_date.strftime('%Y-%m-%d')}
        Expected Delivery: {delivery_date.strftime('%Y-%m-%d')}
        
        Items Ordered:
        {item_desc}
        
        Subtotal: ${amount:,.2f}
        Shipping: ${amount * 0.05:,.2f}
        Total: ${amount * 1.05:,.2f}
        
        Billing Address: 123 Business St, Corporate City, CC 12345
        Shipping Address: Same as billing
        
        Terms: Payment due within 30 days of delivery
        """.strip()
        
        doc = Document(
            id=f"po_{i+1:03d}",
            content=content,
            entity_id=vendor.lower().replace(" ", "_").replace(".", ""),
            doc_type=DocumentType.PURCHASE_ORDER,
            date=po_date,
            metadata={
                "amount": amount * 1.05,
                "shipping_cost": amount * 0.05,
                "delivery_date": delivery_date.isoformat(),
                "item_category": item_desc.split()[0].lower()
            }
        )
        pos.append(doc)
    
    return pos


def generate_contracts(count: int = 10) -> List[Document]:
    """Generate sample contract documents."""
    contractors = [
        "CloudHost Services", "SecureData Inc", "TechSupport Pro",
        "Marketing Experts", "Legal Advisors LLC", "Consulting Group",
        "Maintenance Solutions", "Security Systems", "Training Institute"
    ]
    
    services = [
        ("Cloud hosting and backup services", 199, 999),
        ("IT support and maintenance", 150, 800),
        ("Marketing and advertising services", 500, 3000),
        ("Legal consultation services", 200, 1500),
        ("Business consulting", 300, 2000),
        ("Software maintenance and updates", 100, 600),
        ("Security monitoring services", 250, 1200),
        ("Employee training programs", 400, 2500),
        ("Facility maintenance services", 300, 1800)
    ]
    
    contracts = []
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(count):
        contractor = random.choice(contractors)
        service_desc, min_monthly, max_monthly = random.choice(services)
        monthly_fee = round(random.uniform(min_monthly, max_monthly), 2)
        contract_date = base_date + timedelta(days=random.randint(0, 300))
        duration = random.choice([6, 12, 24, 36])
        end_date = contract_date + timedelta(days=duration * 30)
        
        content = f"""
        SERVICE AGREEMENT #{f'SA-2024-{i+1:03d}'}
        
        Service Provider: {contractor}
        Client: Your Company Name
        
        Contract Date: {contract_date.strftime('%Y-%m-%d')}
        Effective Period: {contract_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
        Duration: {duration} months
        
        Services Description:
        {service_desc}
        
        Financial Terms:
        Monthly Fee: ${monthly_fee:,.2f}
        Total Contract Value: ${monthly_fee * duration:,.2f}
        Payment Schedule: Monthly in advance
        
        Terms and Conditions:
        - Services to be provided as specified in Exhibit A
        - Either party may terminate with 30 days written notice
        - All fees are non-refundable
        - Service level agreements as defined in Exhibit B
        
        This agreement is governed by the laws of the State of California.
        """.strip()
        
        doc = Document(
            id=f"contract_{i+1:03d}",
            content=content,
            entity_id=contractor.lower().replace(" ", "_").replace(".", ""),
            doc_type=DocumentType.CONTRACT,
            date=contract_date,
            metadata={
                "monthly_fee": monthly_fee,
                "duration_months": duration,
                "total_value": monthly_fee * duration,
                "end_date": end_date.isoformat(),
                "service_category": service_desc.split()[0].lower()
            }
        )
        contracts.append(doc)
    
    return contracts


def generate_receipts(count: int = 25) -> List[Document]:
    """Generate sample receipt documents."""
    merchants = [
        "Business Travel Agency", "Hotel California", "Enterprise Car Rental",
        "Airport Parking", "Restaurant Deluxe", "Coffee Shop Express",
        "Gas Station Plus", "Conference Center", "Office Supply Store"
    ]
    
    categories = [
        ("Business travel - airfare", 200, 1200),
        ("Business travel - hotel", 100, 400),
        ("Business travel - car rental", 50, 300),
        ("Business travel - parking", 10, 50),
        ("Business meals - client entertainment", 50, 200),
        ("Business meals - employee lunch", 15, 80),
        ("Transportation - fuel", 30, 100),
        ("Conference and training fees", 100, 800),
        ("Office supplies and materials", 20, 150)
    ]
    
    receipts = []
    base_date = datetime.now() - timedelta(days=90)
    
    for i in range(count):
        merchant = random.choice(merchants)
        category_desc, min_amount, max_amount = random.choice(categories)
        amount = round(random.uniform(min_amount, max_amount), 2)
        receipt_date = base_date + timedelta(days=random.randint(0, 90))
        
        content = f"""
        RECEIPT #{f'RCP-{i+1:04d}'}
        
        Merchant: {merchant}
        Date: {receipt_date.strftime('%Y-%m-%d')}
        Time: {receipt_date.strftime('%H:%M')}
        
        Description: {category_desc}
        Amount: ${amount:,.2f}
        Tax: ${amount * 0.0875:,.2f}
        Total: ${amount * 1.0875:,.2f}
        
        Payment Method: Corporate Credit Card
        Card Last 4 Digits: {random.randint(1000, 9999)}
        
        Business Purpose: {category_desc.split(' - ')[0].title()}
        Employee: John Doe
        Department: Operations
        """.strip()
        
        doc = Document(
            id=f"receipt_{i+1:03d}",
            content=content,
            entity_id=merchant.lower().replace(" ", "_").replace(".", ""),
            doc_type=DocumentType.RECEIPT,
            date=receipt_date,
            metadata={
                "amount": amount * 1.0875,
                "tax_rate": 0.0875,
                "category": category_desc.split(' - ')[0].lower().replace(' ', '_'),
                "employee": "john_doe",
                "department": "operations"
            }
        )
        receipts.append(doc)
    
    return receipts


def save_sample_documents(output_dir: Path):
    """Generate and save all sample documents."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate documents
    invoices = generate_invoices(20)
    purchase_orders = generate_purchase_orders(15)
    contracts = generate_contracts(10)
    receipts = generate_receipts(25)
    
    all_documents = invoices + purchase_orders + contracts + receipts
    
    # Save as JSON for easy loading
    documents_data = []
    for doc in all_documents:
        documents_data.append({
            "id": doc.id,
            "content": doc.content,
            "entity_id": doc.entity_id,
            "doc_type": doc.doc_type.value,
            "date": doc.date.isoformat(),
            "metadata": doc.metadata
        })
    
    with open(output_dir / "sample_documents.json", "w") as f:
        json.dump(documents_data, f, indent=2)
    
    # Save by document type
    for doc_type, docs in [
        ("invoices", invoices),
        ("purchase_orders", purchase_orders),
        ("contracts", contracts),
        ("receipts", receipts)
    ]:
        type_data = []
        for doc in docs:
            type_data.append({
                "id": doc.id,
                "content": doc.content,
                "entity_id": doc.entity_id,
                "doc_type": doc.doc_type.value,
                "date": doc.date.isoformat(),
                "metadata": doc.metadata
            })
        
        with open(output_dir / f"{doc_type}.json", "w") as f:
            json.dump(type_data, f, indent=2)
    
    print(f"Generated {len(all_documents)} sample documents:")
    print(f"  - {len(invoices)} invoices")
    print(f"  - {len(purchase_orders)} purchase orders")
    print(f"  - {len(contracts)} contracts")
    print(f"  - {len(receipts)} receipts")
    print(f"Saved to: {output_dir}")
    
    return all_documents


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    save_sample_documents(output_dir)
