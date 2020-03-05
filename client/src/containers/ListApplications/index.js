import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import Papersheet, {
  DemoWrapper,
} from '../../components/utility/papersheet';
import { Row, FullColumn } from '../../components/utility/rowColumn';
import applicationActions from "../../redux/ListApplications/actions";
import {connect} from "react-redux";
import {store} from "../../redux/store";

import BasicTable from './basicTable';
import {Root, Table} from "./materialUiTables.style";
import {TableBody, TableCell, TableHead, TableRow} from "../../components/uielements/table";
import Scrollbars from "../../components/utility/customScrollBar";

const styles = theme => ({
  root: {
    width: '100%',
    marginTop: theme.spacing(3),
    overflowX: 'auto',
  },
  table: {
    minWidth: 700,
  },
  tableWrapper: {
    overflowX: 'auto',
  },
});

class ListApplications extends Component {
  componentDidMount() {
    store.dispatch(applicationActions.requestApplicationList());
  }


  render() {
    return (
      <LayoutWrapper>
        <Row>
          <FullColumn>
            <Papersheet>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Name</TableCell>
                        <TableCell>URL</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>

                      {(this.props.applications || []).map(application => {
                        return (
                            <TableRow key={application._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/applications/${application._id.$oid}`)}>
                              <TableCell>{application.name}</TableCell>
                              <TableCell>{application.url}</TableCell>
                            </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
            </Papersheet>
          </FullColumn>
        </Row>
      </LayoutWrapper>
    );
  }
}

const mapStateToProps = (state) => {return { ...state.ListApplications} };
export default connect(mapStateToProps)(withStyles(styles, { withTheme: true })(ListApplications));
