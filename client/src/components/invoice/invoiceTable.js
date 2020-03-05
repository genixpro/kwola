import React from 'react';
import Table, {
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from '../uielements/table/';
import TextField from '../uielements/textfield';
import Icon from '../uielements/icon';
import { stringToPosetiveInt } from '../../helpers/utility';

const viewColumns = [
  {
    title: '#',
    dataIndex: 'key',
    rowKey: 'key',
  },
  {
    title: 'Item Name',
    dataIndex: 'itemName',
    rowKey: 'itemName',
  },
  {
    title: 'Unit Costs',
    dataIndex: 'costs',
    rowKey: 'costs',
  },
  {
    title: 'Unit',
    dataIndex: 'qty',
    rowKey: 'qty',
  },
  {
    title: 'Price',
    dataIndex: 'price',
    rowKey: 'price',
  },
];
const editColumns = [
  ...viewColumns,
  {
    title: '',
    dataIndex: 'delete',
    rowKey: 'delete',
  },
];
const HeaderRow = ({ columns }) => (
  <TableHead>
    <TableRow>
      {columns.map(column => (
        <TableCell key={column.rowKey}>{column.title}</TableCell>
      ))}
    </TableRow>
  </TableHead>
);
const ViewTable = ({ invoiceList }) => (
  <Table>
    <HeaderRow columns={viewColumns} />
    <TableBody>
      {invoiceList.map(singleInvoice => (
        <TableRow key={singleInvoice.key}>
          {viewColumns.map(column => (
            <TableCell key={column.rowKey}>
              {singleInvoice[column.rowKey]}
            </TableCell>
          ))}
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const EditTable = ({ editableInvoice, editInvoice, updateValues }) => {
  const { invoiceList } = editableInvoice;
  return (
    <Table>
      <HeaderRow columns={editColumns} />
      <TableBody>
        {invoiceList.map((singleInvoice, index) => (
          <TableRow key={singleInvoice.key}>
            <TableCell>{index + 1}</TableCell>
            <TableCell>
              <TextField
                placeholder="Item Name"
                value={singleInvoice.itemName}
                onChange={event => {
                  editableInvoice.invoiceList[index].itemName =
                    event.target.value;
                  editInvoice(editableInvoice);
                }}
              />
            </TableCell>
            <TableCell>
              <TextField
                placeholder="Unit Cost"
                value={singleInvoice.costs}
                onChange={event => {
                  editableInvoice.invoiceList[
                    index
                  ].costs = stringToPosetiveInt(
                    event.target.value,
                    singleInvoice.costs
                  );
                  editInvoice(updateValues(editableInvoice));
                }}
              />
            </TableCell>
            <TableCell>
              <TextField
                placeholder="Units"
                value={singleInvoice.qty}
                onChange={event => {
                  editableInvoice.invoiceList[index].qty = stringToPosetiveInt(
                    event.target.value,
                    singleInvoice.qty
                  );
                  editInvoice(updateValues(editableInvoice));
                }}
              />
            </TableCell>
            <TableCell>{singleInvoice.price}</TableCell>
            <TableCell>
              {invoiceList.length === 1 ? (
                ''
              ) : (
                <Icon
                  onClick={() => {
                    const newInvoiceList = [];
                    invoiceList.forEach((invoice, i) => {
                      if (i !== index) {
                        newInvoiceList.push(invoice);
                      }
                    });
                    editableInvoice.invoiceList = newInvoiceList;
                    editInvoice(updateValues(editableInvoice));
                  }}
                >
                  delete
                </Icon>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};
export { ViewTable, EditTable };
